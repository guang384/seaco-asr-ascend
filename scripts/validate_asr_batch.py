#!/usr/bin/env python3
"""批前向数值门：真模型上验证「同桶凑批 generate」与「逐条 generate」文本一致
（SeACo-Paraformer padding 路径安全性；升级 funasr/torch 后必跑）。

用法（开发机 CPU / 服务器 NPU 均可）：
    python scripts/validate_asr_batch.py [--device cpu] [--batch 4]
    服务器容器内：
    docker exec asr bash scripts/with_ascend_env.sh python3 scripts/validate_asr_batch.py --device npu

退出码：0 全部一致；1 存在不一致或批路异常。
本地实测（2026-11-17，funasr 1.4.2 CPU）：3 条不等长噪声批前向与逐条文本全一致，
批 0.30s vs 逐条 0.93s（3.1x）。
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from asr_common import load_asr, resolve_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="批前向 vs 逐条 数值等价性验证")
    ap.add_argument("--device", default="cpu",
                    help="cpu/npu/cuda/xpu（默认 cpu；npu 会做 fail-fast 探卡）")
    ap.add_argument("--batch", type=int, default=4, help="批宽（默认 4）")
    ap.add_argument("--real", default=None,
                    help="可选：wav 文件目录，用真实语音替代合成噪声")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[validate] device = {device}")
    m = load_asr(device)

    if args.real:
        import soundfile as sf
        waves = []
        for f in sorted(os.listdir(args.real))[:args.batch]:
            if f.lower().endswith((".wav", ".mp3", ".flac")):
                x, fs = sf.read(os.path.join(args.real, f), dtype="float32",
                                always_2d=True)
                waves.append(x.mean(axis=1))
    else:
        rng = np.random.RandomState(0)
        # 不等长低幅噪声（覆盖 padding 路径；真实语音用 --real）
        waves = [(rng.randn(n) * 0.01).astype(np.float32)
                 for n in (16000, 32000, 24000, 8000)[:args.batch]]

    t0 = time.perf_counter()
    seq = [m.generate(input=w, disable_pbar=True)[0].get("text", "")
           for w in waves]
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        bat = m.generate(input=list(waves), batch_size=len(waves),
                         disable_pbar=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 批前向异常: {type(exc).__name__}: {str(exc)[:200]}")
        sys.exit(1)
    t_bat = time.perf_counter() - t0
    texts = [r.get("text", "") for r in bat]

    n_bad = 0
    for i, (a, b) in enumerate(zip(seq, texts)):
        ok = a == b
        n_bad += 0 if ok else 1
        print(f"  [{i}] {'✓' if ok else '≠'} seq={a!r}")
        if not ok:
            print(f"       batch={b!r}")
    print(f"[validate] {len(waves)} 条：{len(waves)-n_bad} 一致 / {n_bad} 不一致")
    print(f"[validate] 耗时 seq={t_seq:.2f}s  batch={t_bat:.2f}s"
          f"（加速 {t_seq/max(t_bat, 1e-9):.1f}x）")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
