#!/usr/bin/env python3
"""ASR 服务压测：从目录随机抽音频，闭环并发打 REST 接口，直至时长/请求数达标。

与 run_batch_http.py 的分工：那个是批量回放（每文件恰好一次，测单轮吞吐）；
本工具是压力发生器（持续**随机抽样**，测饱和容量/时延分布/长时间稳定性）。

用法：
    python scripts/stress_asr.py bench_audio -c 4 -d 60          # 4 并发压 60s
    python scripts/stress_asr.py bench_audio -c 8 -n 200         # 8 并发共 200 请求
    python scripts/stress_asr.py bench_audio -c 4 --rps 2        # 定速 2 请求/秒
    python scripts/stress_asr.py bench_audio -c 4 -d 120 --mode base64 --seed 42

行为：
- 每请求从目录**均匀随机**抽一个文件（--seed 可复现抽样序列）；
- 闭环模型：N 个 worker 各自"发请求→等响应→再抽下一个"，并发数即压力旋钮；
- --rps 定速时全局请求起始间隔 ≥ 1/rps（超过服务能力的部分在客户端排队）；
- 服务不可达时失败退避 0.2s，不空转刷错；
- Ctrl+C 优雅收尾：等在途请求回来，照常出报告。

报告：控制台汇总（吞吐/时延分位/错误分类）+ 逐请求明细 CSV（--out）。
服务端过载的第一信号看 /metrics 的 asr_queue_wait_ms（各队列等待时间）。
"""
import argparse
import base64
import csv
import os
import random
import statistics
import sys
import threading
import time
from collections import Counter
from datetime import datetime

import requests
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from asr_common import AUDIO_EXTS  # noqa: E402

URL = os.getenv("ASR_URL", "http://127.0.0.1:10095/v1/audio/transcriptions")

_stop = threading.Event()       # Ctrl+C / 达标置位：worker 发完在途即收
_lock = threading.Lock()        # 统计与明细写入锁
_stats = Counter()
_latencies = []                 # 成功请求的时延（秒）
_records = []                   # 逐请求明细（CSV 落盘用）


def discover(input_dir: str) -> list:
    """目录递归收集音频 → [(path, duration_s)]；读不了时长的文件告警跳过。"""
    files = []
    for root, _dirs, names in os.walk(input_dir):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in AUDIO_EXTS:
                path = os.path.join(root, name)
                try:
                    files.append((path, sf.info(path).duration))
                except Exception as exc:  # noqa: BLE001  坏文件只跳自身
                    print(f"[warn] 跳过不可读音频 {name}: {str(exc)[:80]}")
    return files


def _pctl(vals: list, p: float):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


class Pacer:
    """可选定速：全局请求起始间隔 ≥ 1/rps（不限速时零开销直通）。"""

    def __init__(self, rps):
        self.interval = 1.0 / rps if rps and rps > 0 else None
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self):
        if self.interval is None:
            return
        with self._lock:
            send_at = max(time.monotonic(), self._next)
            self._next = send_at + self.interval
        time.sleep(max(0.0, send_at - time.monotonic()))


def _build_request(path: bytes, name: str, mode: str, hotword) -> dict:
    """按输入模式构造 requests.post 的 kwargs（file=multipart / base64=JSON）。"""
    if mode == "base64":
        payload = {"audio_base64": base64.b64encode(path).decode(), "filename": name}
        if hotword:
            payload["hotword"] = hotword
        return {"json": payload}
    kw = {"files": {"file": (name, path)}}
    if hotword:
        kw["data"] = {"hotword": hotword}
    return kw


def worker(wid: int, args, files, deadline, claimed, pace, rng) -> None:
    """闭环压测 worker：随机抽文件 → POST → 记账 → 循环，直至达标/停止。"""
    sess = requests.Session()          # 每 worker 独立连接池（Session 非线程安全）
    seq = 0
    while not _stop.is_set():
        if args.requests and claimed[0] >= args.requests:
            return
        if deadline is not None and time.monotonic() >= deadline:
            return
        path, dur = rng.choice(files)
        if args.requests:
            with _lock:                # 计数型停止：先占号再发，不多发不少发
                if claimed[0] >= args.requests:
                    return
                claimed[0] += 1
        pace.wait()                    # 定速排队不计入请求时延
        status, err = None, None
        t0 = time.perf_counter()
        try:
            name = os.path.basename(path)
            with open(path, "rb") as f:
                data = f.read()
            r = sess.post(args.url, timeout=args.timeout,
                          **_build_request(data, name, args.mode, args.hotword))
            status = r.status_code
            if status != 200:
                err = r.text[:120]
        except Exception as exc:  # noqa: BLE001  网络类失败：退避防空转刷错
            err = f"{type(exc).__name__}: {str(exc)[:100]}"
            time.sleep(0.2)
        latency = time.perf_counter() - t0
        ok = status == 200
        with _lock:
            _stats["sent"] += 1
            _stats["ok" if ok else "fail"] += 1
            if status is not None:
                _stats[f"http_{status}"] += 1
            if err:
                _stats[f"err_{err.split(':', 1)[0][:40]}"] += 1
            if ok:
                _stats["audio_s"] += dur
                _latencies.append(latency)
            _records.append({"seq": f"w{wid}-{seq}",
                             "start": datetime.now().isoformat(timespec="seconds"),
                             "file": os.path.basename(path),
                             "audio_s": round(dur, 2),
                             "latency_s": round(latency, 3),
                             "status": status if status is not None else "",
                             "ok": int(ok), "error": err or ""})
        seq += 1


def report(args, files, wall: float) -> int:
    total_audio = sum(d for _p, d in files)
    sent, ok, fails = _stats["sent"], _stats["ok"], _stats["fail"]
    stop = (f"requests={args.requests}" if args.requests
            else f"duration={args.duration:g}s")
    print("\n" + "=" * 62)
    print(f"# ASR 压测报告  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"# 目录: {args.input_dir}  {len(files)} 个文件  共 {total_audio:.0f}s 音频")
    print(f"# 模式: {args.mode}  并发: {args.concurrency}"
          f"  rps: {args.rps or '不限'}  停止: {stop}")
    if wall > 0:
        print(f"# 墙钟 {wall:.1f}s  发送 {sent}  成功 {ok}  失败 {fails}"
              f"  ({sent / wall:.1f} 请求/s)")
    if _latencies:
        print(f"# 时延  avg={statistics.mean(_latencies):.2f}s"
              f"  p50={_pctl(_latencies, .5):.2f}s"
              f"  p90={_pctl(_latencies, .9):.2f}s"
              f"  p95={_pctl(_latencies, .95):.2f}s"
              f"  max={max(_latencies):.2f}s")
        if wall > 0:
            print(f"# 音频吞吐  {_stats['audio_s'] / wall:.1f}x 实时"
                  f"  （成功累计 {_stats['audio_s']:.0f}s 音频 / 墙钟 {wall:.1f}s）")
    http_line = "  ".join(f"{k[5:]}:{v}" for k, v in sorted(_stats.items())
                          if k.startswith("http_"))
    if http_line:
        print(f"# HTTP 分布  {http_line}")
    err_line = "  ".join(f"{k[4:]}:{v}" for k, v in sorted(_stats.items())
                         if k.startswith("err_"))
    if err_line:
        print(f"# 异常分布  {err_line}")
    print("=" * 62)
    # 逐请求明细 CSV（分析时延-文件长度相关性 / 复盘失败用）
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seq", "start", "file", "audio_s",
                                          "latency_s", "status", "ok", "error"])
        w.writeheader()
        w.writerows(_records)
    print(f"[done] 逐请求明细已写入 {args.out}")
    if sent and ok == 0:
        print("[error] 全部请求失败——检查服务是否已启动 / URL 是否正确")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description="ASR 随机抽样压测（闭环并发）")
    ap.add_argument("input_dir", help="音频目录（递归收集，随机抽样）")
    ap.add_argument("-c", "--concurrency", type=int, default=4,
                    help="并发 worker 数（默认 4）")
    ap.add_argument("-d", "--duration", type=float, default=None,
                    help="压测时长秒（与 -n 二选一；都不给默认 60）")
    ap.add_argument("-n", "--requests", type=int, default=None,
                    help="总请求数（与 -d 二选一）")
    ap.add_argument("--rps", type=float, default=None,
                    help="定速上限（请求/秒；默认不限）")
    ap.add_argument("--mode", choices=["file", "base64"], default="file",
                    help="输入模式：multipart file / JSON audio_base64（默认 file）")
    ap.add_argument("--hotword", default=None, help="热词（逗号分隔，随请求携带）")
    ap.add_argument("--timeout", type=float, default=300,
                    help="单请求超时秒（默认 300）")
    ap.add_argument("--url", default=URL, help=f"服务地址（默认 {URL}）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（复现抽样序列）")
    ap.add_argument("--out", default="stress_report.csv",
                    help="逐请求明细 CSV（默认 stress_report.csv）")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"[error] 目录不存在: {args.input_dir}")
    files = discover(args.input_dir)
    if not files:
        raise SystemExit(f"[error] 目录里没有可读音频: {args.input_dir}")
    if not args.requests and args.duration is None:
        args.duration = 60.0

    stop = (f"requests={args.requests}" if args.requests
            else f"duration={args.duration:g}s")
    print(f"[info] {len(files)} 个文件 共 {sum(d for _p, d in files):.0f}s 音频，"
          f"并发 {args.concurrency}，模式 {args.mode}，停止条件 {stop}")

    deadline = (time.monotonic() + args.duration) if args.duration else None
    claimed = [0]
    pace = Pacer(args.rps)
    threads = [threading.Thread(
        target=worker, daemon=True,
        args=(i, args, files, deadline, claimed, pace,
              random.Random(None if args.seed is None else args.seed + i)))
        for i in range(args.concurrency)]
    t_start = time.monotonic()
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            line = (f"[{time.monotonic() - t_start:6.0f}s] "
                    f"sent={_stats['sent']} ok={_stats['ok']} "
                    f"fail={_stats['fail']}")
            if _latencies:
                line += f"  avg={statistics.mean(_latencies):.2f}s"
            print(line, flush=True)
    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C——等在途请求收尾…", flush=True)
        _stop.set()
    for t in threads:
        t.join()
    return report(args, files, time.monotonic() - t_start)


if __name__ == "__main__":
    sys.exit(main())
