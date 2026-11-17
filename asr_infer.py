#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FunASR 本地推理脚本：SeaCo-Paraformer(ASR) + FSMN-VAD + CT-Transformer(标点)

- 本地优先：直接加载已下载到工作目录的三个模型，不联网下载
- 设备可配置：CPU / NPU（需 torch_npu）/ XPU / CUDA
- 用法：
    本地 CPU：  python asr_infer.py --audio xxx.wav
    指定设备：  python asr_infer.py --audio xxx.wav --device cpu
    NPU 冒烟：  python asr_infer.py --audio xxx.wav --device npu
    热词：      python asr_infer.py --audio xxx.wav --hotword "魔搭,达摩院"
"""
import argparse
import os
import time

from asr_common import ASR_DIR, PUNC_DIR, VAD_DIR, resolve_device


def main():
    ap = argparse.ArgumentParser(description="FunASR 本地 ASR 推理（ASR+VAD+标点）")
    ap.add_argument("--asr", default=ASR_DIR, help="ASR 模型目录")
    ap.add_argument("--vad", default=VAD_DIR, help="VAD 模型目录")
    ap.add_argument("--punc", default=PUNC_DIR, help="标点模型目录")
    ap.add_argument("--audio", default=os.path.join(ASR_DIR, "asr_example_hotword.wav"),
                    help="输入音频路径(16k wav)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "npu", "xpu"], help="推理设备")
    ap.add_argument("--hotword", default=None, help="热词，逗号分隔，如 '魔搭,达摩院'")
    ap.add_argument("--no-punc", action="store_true", help="关闭标点模型")
    ap.add_argument("--no-vad", action="store_true", help="关闭 VAD（短音频可关）")
    args = ap.parse_args()

    # 本地模型目录必须存在
    for name, path in [("asr", args.asr), ("vad", args.vad), ("punc", args.punc)]:
        if (name == "vad" and args.no_vad) or (name == "punc" and args.no_punc):
            continue
        if not os.path.isdir(path):
            raise SystemExit(f"[error] {name} 模型目录不存在: {path}")

    device = resolve_device(args.device)
    print(f"[info] device = {device}")

    from funasr import AutoModel
    from asr_common import base_model_kw
    # 与 compute worker 同一 AutoModel 参数口径（ngpu 只对 cpu 传 0——
    # funasr 内部 ngpu=0 会强制回退 CPU，见 asr_common.base_model_kw）
    kw = base_model_kw(device)
    kw["model"] = args.asr
    if not args.no_vad:
        kw["vad_model"] = args.vad
    if not args.no_punc:
        kw["punc_model"] = args.punc

    t0 = time.time()
    model = AutoModel(**kw)
    print(f"[info] 模型加载耗时 {time.time() - t0:.1f}s")

    gen_kw = {}
    if args.hotword:
        gen_kw["hotword"] = args.hotword
    t1 = time.time()
    res = model.generate(input=args.audio, **gen_kw)
    print(f"[info] 推理耗时 {time.time() - t1:.1f}s")
    print("-" * 40)
    for r in res:
        # r 一般含 key/text/timestamp 等字段
        text = r.get("text", "") if isinstance(r, dict) else str(r)
        print(f"[{r.get('key', '') if isinstance(r, dict) else ''}] {text}")


if __name__ == "__main__":
    main()
