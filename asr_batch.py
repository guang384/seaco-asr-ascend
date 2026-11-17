#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量离线转写（吞吐优先）

- 模型只加载一次，FunASR 内部 VAD→ASR→标点 全流水线
- 多文件按 batch_size_s（单批语音秒数）动态组批
- 输出：每音频一个 .txt + 汇总 stats.json（RTF / 吞吐倍速）

用法：
    python asr_batch.py --input 目录或文件... --out transcripts --device xpu
    调组批粒度：--batch-size-s 30（默认；iGPU 实测 30s 最优 31x，300s 仅 11.5x，NPU 需重测）
"""
import argparse
import glob
import json
import os
import time

from asr_common import AUDIO_EXTS

# 目录递归收集用的 glob 形态（由 AUDIO_EXTS 派生，扩展名口径不漂移）
AUDIO_GLOBS = tuple(f"*{ext}" for ext in AUDIO_EXTS)


def collect(inputs: list[str]) -> list[str]:
    """展开目录（递归）与通配符，收集音频文件"""
    files = []
    for path in inputs:
        path = os.path.normpath(path)
        if os.path.isdir(path):
            for ext in AUDIO_GLOBS:
                files.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))
        else:
            files.extend(glob.glob(path))
    # 去重保序
    seen, out = set(), []
    for f in files:
        f = os.path.abspath(f)
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main():
    import soundfile as sf
    from asr_common import load_offline_model, resolve_device

    ap = argparse.ArgumentParser(description="批量 ASR 转写（吞吐优先）")
    ap.add_argument("--input", nargs="+", required=True,
                    help="音频文件/目录/通配符，可多个")
    ap.add_argument("--out", default="transcripts", help="输出目录")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "npu", "xpu"], help="推理设备")
    ap.add_argument("--batch-size-s", type=int, default=30,
                    help="单批语音秒数（默认 30；iGPU 实测小批更高效，NPU 需重测）")
    ap.add_argument("--hotword", default=None, help="热词，逗号分隔")
    args = ap.parse_args()

    files = collect(args.input)
    if not files:
        raise SystemExit(f"[error] 未找到音频文件: {args.input}")
    audio_total = sum(sf.info(f).duration for f in files)

    device = resolve_device(args.device)
    print(f"[info] device = {device}，{len(files)} 个文件，共 {audio_total:.1f}s 音频")

    t0 = time.time()
    model = load_offline_model(device)
    print(f"[info] 模型加载 {time.time() - t0:.1f}s")

    gen_kw = dict(batch_size_s=args.batch_size_s)
    if args.hotword:
        gen_kw["hotword"] = args.hotword

    t1 = time.time()
    results = model.generate(input=files, **gen_kw)
    infer_s = time.time() - t1

    os.makedirs(args.out, exist_ok=True)
    # funasr 返回的 key 是文件名去扩展名（stem），按 stem 匹配
    by_key = {}
    for r in results:
        if isinstance(r, dict) and r.get("key"):
            by_key[os.path.splitext(os.path.basename(r["key"]))[0]] = r
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        text = by_key.get(stem, {}).get("text", "")
        with open(os.path.join(args.out, stem + ".txt"), "w", encoding="utf-8") as fp:
            fp.write(text)
        print(f"  [ok] {os.path.basename(f)} -> {text[:60]}")

    wall = time.time() - t0
    rtf = infer_s / audio_total
    stats = dict(n_files=len(files), audio_seconds=round(audio_total, 2),
                 model_load_s=round(time.time() - t0 - infer_s, 2),
                 infer_s=round(infer_s, 2), wall_s=round(wall, 2),
                 rtf=round(rtf, 4), realtime_speed=f"{1 / rtf:.1f}x",
                 device=device, batch_size_s=args.batch_size_s)
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    print("-" * 50)
    print(f"[info] 推理 {infer_s:.2f}s / 音频 {audio_total:.1f}s → RTF {rtf:.3f}"
          f"（{1 / rtf:.1f}x 实时，含加载总耗时 {wall:.2f}s）")


if __name__ == "__main__":
    main()
