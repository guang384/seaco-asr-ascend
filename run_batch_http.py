"""HTTP 批量转写（并发版）：线程池 N 路同时 POST，结果写 output.txt

用法：python run_batch_http.py [目录] [输出文件] [并发数] [file|base64]
记录：每请求开始/结束时间、耗时（并发下含服务端排队）、响应 JSON
汇总：墙钟总耗时、吞吐倍速、单请求耗时 p50/p90/max
file   = multipart 文件上传（OpenAI 兼容口径）
base64 = JSON {"audio_base64"}（双模式输入的第二种，压测两条链路等价性）
"""
import base64
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests
import soundfile as sf

from asr_common import AUDIO_EXTS

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "录音样本")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "output.txt")
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
INPUT_MODE = sys.argv[4] if len(sys.argv) > 4 else "file"
URL = os.getenv("ASR_URL", "http://127.0.0.1:10095/v1/audio/transcriptions")

files = sorted(p for p in DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS)
total_audio = sum(sf.info(str(f)).duration for f in files)
print(f"[info] {len(files)} 个文件，总时长 {total_audio:.0f}s，并发 {WORKERS}，输入 {INPUT_MODE}")

sess = requests.Session()


def run_one(i: int, f: Path) -> tuple:
    dur = sf.info(str(f)).duration
    data = f.read_bytes()
    t_start = datetime.now()
    t0 = time.perf_counter()
    try:
        if INPUT_MODE == "base64":
            r = sess.post(URL, json={"audio_base64": base64.b64encode(data).decode(),
                                     "filename": f.name}, timeout=1800)
        else:
            r = sess.post(URL, files={"file": (f.name, data)}, timeout=1800)
        cost = time.perf_counter() - t0
        t_end = datetime.now()
        rec = (f"[{i:2d}/{len(files)}] {f.name}  {dur:.1f}s音频\n"
               f"  开始 {t_start:%H:%M:%S.%f}  结束 {t_end:%H:%M:%S.%f}  耗时 {cost:.2f}s  HTTP {r.status_code}\n"
               f"  {r.text.strip()}")
        print(f"[{i:2d}/{len(files)}] {f.name}  {dur:.1f}s  {cost:.1f}s  HTTP {r.status_code}")
        return rec, cost
    except Exception as e:
        cost = time.perf_counter() - t0
        t_end = datetime.now()
        print(f"[{i:2d}/{len(files)}] {f.name}  异常 {cost:.1f}s: {e}")
        return (f"[{i:2d}/{len(files)}] {f.name}  {dur:.1f}s音频\n"
                f"  开始 {t_start:%H:%M:%S.%f}  结束 {t_end:%H:%M:%S.%f}  耗时 {cost:.2f}s  请求异常: {e}", cost)


t_all = time.perf_counter()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    results = list(pool.map(run_one, range(1, len(files) + 1), files))
wall = time.perf_counter() - t_all
records = [r for r, _ in results]
costs = sorted(c for _, c in results)
p50 = statistics.median(costs)
p90 = costs[int(len(costs) * 0.9) - 1]

header = [
    f"# HTTP 批量转写记录（并发 {WORKERS} 路，输入 {INPUT_MODE}）  {datetime.now():%Y-%m-%d %H:%M:%S}",
    f"# 目录: {DIR}  文件数: {len(files)}  总音频: {total_audio:.0f}s",
    f"# 墙钟 {wall:.1f}s  吞吐 {total_audio / wall:.1f}x 实时  "
    f"单请求耗时 p50={p50:.2f}s p90={p90:.2f}s max={costs[-1]:.2f}s（含服务端排队）",
    ""]
OUT.write_text("\n".join(header + records) + "\n", encoding="utf-8")
print(f"[done] 墙钟 {wall:.1f}s  吞吐 {total_audio / wall:.1f}x  "
      f"p50={p50:.2f}s p90={p90:.2f}s max={costs[-1]:.2f}s  已写入 {OUT}")
