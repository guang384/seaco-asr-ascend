#!/usr/bin/env python3
"""benchmark 音频集构建：中文会议长录音（AISHELL-4）截取 50 个**不同长度**区间。

语料来源：AISHELL-4 官方 HF 仓库（hf-mirror 镜像）真实中文多人会议录音
（每场 30 分钟~数小时，含自然停顿/抢话/串音——VAD 与长音频链路的真实负载）。
从若干场会议中按黄金比散布截取区间，产出 50 个不同长度的 wav（1s~10min），
覆盖短句 / 中段 / 长段全谱。

用法（开发机，需 requests + soundfile + numpy）：
    python scripts/build_bench_corpus.py                  # 联网下载会议 + 截取 50 个
    python scripts/build_bench_corpus.py --cut-from 目录/  # 从本地录音截取（涉密数据不出机）
    python scripts/build_bench_corpus.py 30               # 自定义数量

合规说明：
- 公开语料为 AISHELL-4（CC BY-4.0/科研用途）；--cut-from 用于自己的录音，
  全部处理在本机完成、不外传；
- bench_audio/ 与一切音频被 .dockerignore 全局排除——**绝不打进镜像**。
"""
import os
import sys
import time

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from asr_common import AUDIO_EXTS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "bench_audio"))
SRC = os.path.join(OUT, "_sources")

# AISHELL-4 官方会议场次（hf-mirror 镜像；L/M/S = 远/中/近场，C = 通道）
AISHELL4 = "https://hf-mirror.com/datasets/AISHELL/AISHELL-4/resolve/main/test/wav/{name}"
DEFAULT_SOURCES = [
    AISHELL4.format(name="S_R003S01C01.flac"),   # 近场，语音清晰
    AISHELL4.format(name="M_R003S02C01.flac"),   # 中场
    AISHELL4.format(name="L_R003S02C02.flac"),   # 远场，含混响串音
]

LENGTHS_HINT = None   # None = 生成 50 个不同长度（对数均匀 1s~600s）


def gen_lengths(n: int) -> list:
    """n 个不同长度：对数均匀铺满 1s~600s（短句密、长段疏，贴近真实请求分布）。"""
    rng = np.random.default_rng(20261116)
    vals = np.exp(rng.uniform(np.log(1.0), np.log(600.0), size=n * 3))
    out, seen = [], set()
    for v in vals:
        d = round(float(v), 1)
        if d not in seen and 1.0 <= d <= 600.0:
            seen.add(d)
            out.append(d)
        if len(out) >= n:
            break
    return sorted(out)


def download(urls: list) -> list:
    os.makedirs(SRC, exist_ok=True)
    files = []
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        path = os.path.join(SRC, name)
        if os.path.exists(path) and os.path.getsize(path) > 10_000_000:
            print(f"[src] 已有 {name}（{os.path.getsize(path)/2**20:.0f}MB）")
            files.append(path)
            continue
        print(f"[src] 下载 {name} …")
        t0 = time.time()
        try:
            with requests.get(url, stream=True, timeout=(15, 300)) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        except Exception as exc:  # noqa: BLE001  单源失败跳过（其余源仍可用）
            print(f"[src]   ✗ {name}: {str(exc)[:80]}（跳过）")
            continue
        print(f"[src]   ✓ {os.path.getsize(path)/2**20:.0f}MB / {time.time()-t0:.0f}s")
        files.append(path)
    return files


def cut_intervals(sources: list, n: int) -> list:
    """从长录音截取 n 个不同长度区间 → bench_audio/（16k mono PCM16 wav）。"""
    infos = [(p, sf.info(p)) for p in sources]
    for p, i in infos:
        print(f"[cut] 源 {os.path.basename(p)}: {i.duration/60:.1f}min @ {i.samplerate}Hz")
    lengths = gen_lengths(n)
    # 源时长上限约束：区间必须放得下（截 95% 以内的头部起步）
    max_len = min(i.duration for _p, i in infos) * 0.95
    lengths = [d for d in lengths if d <= max_len]
    while len(lengths) < n:                      # 超长源不足时补重复区间长度
        lengths.append(round(max_len - len(lengths) * 0.1, 1))
    records = []
    for k, dur in enumerate(lengths):
        path, info = infos[k % len(infos)]
        max_off = info.duration - dur - 1.0
        off = float((k * 0.6180339887) % 1.0) * max(off_min := 0.0, max_off)
        with sf.SoundFile(path) as f:
            f.seek(int(off * info.samplerate))
            audio = f.read(int(dur * info.samplerate), dtype="float32",
                           always_2d=True)
        audio = audio.mean(axis=1)
        name = f"bench_{k+1:02d}_{dur:g}s.wav"
        sf.write(os.path.join(OUT, name), audio, info.samplerate,
                 subtype="PCM_16")
        records.append((name, dur, os.path.basename(path),
                        f"{off:.1f}s~{off+dur:.1f}s"))
        if (k + 1) % 10 == 0:
            print(f"[cut]   {k+1}/{n} …")
    return records


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 50
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):                    # 清掉旧的产物（保留 _sources）
        if f.startswith("bench_") and f.endswith(".wav"):
            os.unlink(os.path.join(OUT, f))

    if "--cut-from" in sys.argv:                 # 涉密/私有录音：本地截取
        i = sys.argv.index("--cut-from")
        roots = sys.argv[i + 1:] or ["."]
        sources = []
        for r in roots:
            if os.path.isfile(r):
                sources.append(r)
            else:
                sources += [os.path.join(r, f) for f in sorted(os.listdir(r))
                            if f.lower().endswith(AUDIO_EXTS)]
        sources = [s for s in sources if os.path.getsize(s) > 100_000]
    else:                                        # 公开会议语料
        sources = download(DEFAULT_SOURCES)

    assert sources, "无可用源音频"
    records = cut_intervals(sources, n)
    assert len(records) == n, f"数量不足 {len(records)}/{n}"
    durs = [r[1] for r in records]
    assert len(set(durs)) == n, "存在重复长度"

    records.sort(key=lambda r: r[1])
    lines = [f"# bench_audio 压测集（{n} 个不同长度，会议/对话长录音截取）",
             "",
             "语料：AISHELL-4 官方中文多人会议录音（真实会议：自然停顿/抢话/串音），",
             "按黄金比散布截取不同区间。**全部音频不打进镜像**（.dockerignore 全局排除）。",
             "",
             "| # | 文件 | 时长(s) | 源录音 | 区间 |", "|---|---|---|---|---|"]
    for i, (name, dur, src, seg) in enumerate(records, 1):
        lines.append(f"| {i} | {name} | {dur} | {src} | {seg} |")
    total = sum(durs)
    lines += ["", f"合计 {n} 个文件，总时长 {total:.0f}s（{total/60:.1f}min）。",
              "",
              "重建：`python scripts/build_bench_corpus.py 50`（联网重新下载会议源）；",
              "用你自己的录音（不出机）：`python scripts/build_bench_corpus.py --cut-from 目录/`。"]
    with open(os.path.join(OUT, "MANIFEST.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] {n} 个文件，总时长 {total:.0f}s（{total/60:.1f}min）→ {OUT}")


if __name__ == "__main__":
    main()
