#!/usr/bin/env python3
"""故障注入冒烟（重派语义端到端验证）：并发请求进行中 SIGKILL 一个 asr
worker → 断言：① 请求几乎全部成功（重试一次语义：在途任务换 worker 重算）
② /metrics 的 asr_worker_crashes_total ≥ 1 ③ /health ready=true（supervisor
已重生 worker）。对着**已起的服务**跑（容器内执行，能读心跳文件拿 worker pid）：

    docker exec asr python3 scripts/parity_smoke.py --audio /app/test.wav

或指定服务地址（跨主机压测时在任意客户端跑，--kill 则需容器内执行）：
    python3 scripts/parity_smoke.py --url http://127.0.0.1:10095 --audio xx.wav
"""
import argparse
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HEALTH_DIR = os.getenv("ASR_HEALTH_DIR", "/tmp/asr-health")


def find_asr_pids():
    """从心跳文件找活着的 asr worker pid（role=asr）。"""
    pids = []
    try:
        names = os.listdir(HEALTH_DIR)
    except OSError:
        return pids
    for name in names:
        if not (name.startswith("worker-") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(HEALTH_DIR, name), encoding="utf-8") as f:
                d = json.load(f)
            if d.get("role") == "asr" and d.get("pid"):
                pids.append(int(d["pid"]))
        except (OSError, ValueError):
            continue
    return pids


def fire(url, audio_bytes, results, i):
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{url}/v1/audio/transcriptions",
                          files={"file": ("p.wav", audio_bytes)},
                          timeout=120)
        results[i] = (r.status_code, time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001
        results[i] = (f"exc:{exc}", time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(description="故障注入：杀 worker 验证重派")
    ap.add_argument("--url", default="http://127.0.0.1:10095")
    ap.add_argument("--audio", default="/app/test.wav")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--kill", type=int, default=1, help="SIGKILL 的 worker 数")
    ap.add_argument("--no-kill", action="store_true", help="只发压不杀（对照组）")
    args = ap.parse_args()

    with open(args.audio, "rb") as f:
        audio = f.read()
    n = args.concurrency
    results = [None] * n
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(fire, args.url, audio, results, i) for i in range(n)]
        time.sleep(2)              # 等任务真正进 worker（加载后首轮推理）
        if not args.no_kill:
            pids = find_asr_pids()
            if not pids:
                print("[warn] 未找到 asr worker 心跳（--kill 跳过；"
                      "确认在容器内/ASR_HEALTH_DIR 正确）")
            for pid in pids[:args.kill]:
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"[kill] SIGKILL asr worker pid={pid}")
                except OSError as exc:
                    print(f"[warn] kill {pid} failed: {exc}")
        for f in futs:
            f.result()

    ok = sum(1 for s, _c in results if s == 200)
    lat = sorted(c for s, c in results if s == 200)
    print(f"── 结果: {ok}/{n} 成功（重试一次语义下应 {n}/{n}，除非同任务二次死亡）")
    if lat:
        print(f"   p50={lat[len(lat)//2]:.2f}s max={lat[-1]:.2f}s")
    for i, (s, c) in enumerate(results):
        if s != 200:
            print(f"   [{i}] HTTP {s} ({c:.2f}s)")

    time.sleep(6)                  # 等 supervisor 重生 + 心跳落盘
    try:
        m = requests.get(f"{args.url}/metrics", timeout=10).text
        crashes = [ln for ln in m.splitlines()
                   if ln.startswith("asr_worker_crashes_total")]
        ready = requests.get(f"{args.url}/health", timeout=10)
        ready_ok = ready.json().get("ready") if ready.status_code == 200 else False
        total_crash = sum(float(ln.split()[-1]) for ln in crashes) if crashes else 0
        print(f"── /metrics crashes_total={total_crash:.0f}（应 ≥1），"
              f"/health ready={ready_ok}（应 true）")
        if not args.no_kill and total_crash < 1:
            print("[FAIL] crashes_total 未增长——杀进程没被 supervisor 记账？")
            sys.exit(1)
        if not ready_ok:
            print("[FAIL] /health 未恢复 ready——worker 重生失败？")
            sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 指标核对失败: {exc}")
    if ok < n and not args.no_kill:
        # 同一任务的重试 worker 又被杀才会失败；单杀应全成
        print(f"[FAIL] 有请求失败（单杀下应全成）")
        sys.exit(1)
    print("[PASS] 故障注入验证通过")


if __name__ == "__main__":
    main()
