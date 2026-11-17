#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公共模块：设备探测 + 本地模型加载（compute / asr_infer / asr_batch 共用）。

设备语义：
- auto（默认）：NPU → XPU → CUDA → CPU 逐级**真探卡**，每级失败记 WARNING
  后降级——降级必须看得见（funasr AutoModel 会静默回退 CPU，把卡坏伪装成
  健康；这里用显式日志消掉该陷阱），**永不退出**。
- 显式 DEVICE=npu：torch_npu 缺失 / 无卡 / 探卡失败 → 直接退出带 remediation
  （明确要 NPU 却拿不到，静默降级是灾难）。
- 显式 cpu / cuda / xpu：按指定执行，不重探。
"""
import logging
import os
import subprocess
import sys

_log = logging.getLogger("asr.device")

HERE = os.path.dirname(os.path.abspath(__file__))
ASR_DIR = os.path.join(HERE, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
VAD_DIR = os.path.join(HERE, "speech_fsmn_vad_zh-cn-16k-common-pytorch")
PUNC_DIR = os.path.join(HERE, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")

# 音频扩展名单一口径（asr_batch / run_batch_http / build_bench_corpus 共用）
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a")

_REMEDIATION = (
    "排查：npu-smi info 看卡是否可见；容器需挂载 /dev/davinci* 与 "
    "/usr/local/Ascend/driver（见 deploy.sh npu_mounts）；检查 "
    "ASCEND_DEVICE_ID；确需 CPU 请显式 --device cpu / DEVICE=cpu")


def _npu_probe(idx: int = 0) -> None:
    """真实探卡：torch.zeros + synchronize（is_available 只看注册表，不代表卡能用）。"""
    import torch
    dev = torch.device(f"npu:{idx}")
    torch.zeros(1, device=dev)
    torch.npu.synchronize(dev)


def pick_npu_index() -> int:
    """NPU 卡号挑选：显式 ASCEND_DEVICE_ID > torch 枚举 pid 打散 > npu-smi 兜底。"""
    import torch
    raw = (os.getenv("ASCEND_DEVICE_ID") or "").strip()
    if raw.isdigit():
        return int(raw)   # 显式指定不因越界丢弃（vNPU 池化下枚举数与真实标识不一致是常态）
    try:
        n = torch.npu.device_count()
        if n > 0:
            return os.getpid() % n   # 多 worker 自动分摊多卡
    except Exception:  # noqa: BLE001
        pass
    # npu-smi 驱动层兜底（torch 探测全败时的真值来源）
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True,
                             text=True, timeout=10).stdout
        ids = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit() and parts[1].isdigit():
                ids.append(int(parts[1]))
        if ids:
            return ids[os.getpid() % len(ids)]
    except Exception:  # noqa: BLE001
        pass
    return 0


def _probe_npu() -> tuple[str, str | None]:
    """NPU 选卡 + 真实探卡（npu 显式 / auto 两分支共用）。
    成功返回 ("npu:idx", None)；失败返回 ("", 原因描述)——由调用方决定
    退出（显式 npu）还是降级（auto）。"""
    idx = pick_npu_index()
    try:
        _npu_probe(idx)
        return f"npu:{idx}", None
    except Exception as exc:  # noqa: BLE001
        return "", f"NPU 卡 {idx} 探测失败: {exc}"


def resolve_device(prefer: str) -> str:
    """解析推理设备。返回 "npu:idx" / "xpu" / "cuda" / "cpu"。

    prefer=auto（默认）：NPU → XPU → CUDA → CPU 逐级探测降级，每级失败
    记 WARNING（降级必须看得见），**永不退出**。
    prefer=npu（显式）：torch_npu 缺失 / 无卡 / 探卡失败 → SystemExit 带
    remediation——明确要 NPU 却拿不到，静默降级是灾难。
    prefer=cpu/cuda/xpu：按指定执行，不重探。"""
    if prefer == "cpu":
        return "cpu"
    import torch
    if prefer == "npu":
        try:
            import torch_npu  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"[fatal] --device npu 但 torch_npu 不可用: {exc}\n{_REMEDIATION}")
        if not torch.npu.is_available():
            sys.exit(f"[fatal] --device npu 但无可用 NPU 卡\n{_REMEDIATION}")
        dev, err = _probe_npu()
        if err:
            sys.exit(f"[fatal] --device npu 但 {err}\n{_REMEDIATION}")
        return dev
    if prefer == "auto":
        # ① NPU 档（真探卡，不只看注册表）
        try:
            import torch_npu  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            _log.warning("auto: torch_npu 不可导入（%s）——跳过 NPU 档", str(exc)[:120])
        else:
            if not torch.npu.is_available():
                _log.warning("auto: torch_npu 可导入但无可用 NPU 卡——降级继续"
                             "（要锁定 NPU 请显式 DEVICE=npu）")
            else:
                dev, err = _probe_npu()
                if err:
                    _log.warning("auto: %s——降级继续（要锁定 NPU 请显式 DEVICE=npu）", err)
                else:
                    _log.info("auto: NPU 探卡通过 → %s", dev)
                    return dev
        # ② XPU 档
        try:
            if torch.xpu.is_available():
                _log.info("auto: → xpu")
                return "xpu"
        except Exception as exc:  # noqa: BLE001
            _log.warning("auto: XPU 探测异常（%s）——继续", exc)
        # ③ CUDA 档
        try:
            if torch.cuda.is_available():
                _log.info("auto: → cuda")
                return "cuda"
        except Exception as exc:  # noqa: BLE001
            _log.warning("auto: CUDA 探测异常（%s）——继续", exc)
        # ④ CPU 兜底
        _log.warning("auto: NPU/XPU/CUDA 均不可用——落到 CPU（要锁卡请显式 DEVICE=npu 等）")
        return "cpu"
    # 显式 cuda/xpu：原样返回（开发便利，不做重探）
    return prefer


def base_model_kw(device: str) -> dict:
    """AutoModel 公共参数：本地模型跳过 hub 更新

    注意：funasr 内部 ngpu=0 会强制回退 CPU（auto_model.py 设备判定），
    所以只有 cpu 设备才传 ngpu=0；xpu/npu/cuda（含带 :idx 的实例设备）保持默认。
    """
    kw = dict(device=device, disable_update=True)
    if device.split(":")[0] == "cpu":
        kw["ngpu"] = 0
    return kw


def load_offline_model(device: str):
    """批量/单文件模式：ASR+VAD+标点 一体（VAD 全量切分，吞吐优先）"""
    from funasr import AutoModel
    return AutoModel(model=ASR_DIR, vad_model=VAD_DIR, punc_model=PUNC_DIR,
                     **base_model_kw(device))


# ---------------- 单件加载：每 worker 进程一份（compute 用） ----------------
# ASR 是唯一瓶颈，按 worker 数复制（每进程一份）；VAD/标点为独立 CPU
# worker 池（吞吐富余 300x/1800x 实时，在 CPU 上不占显存）。

def load_asr(device: str):
    """单 ASR 实例（不带 VAD/标点，编排由调用方负责）"""
    from funasr import AutoModel
    return AutoModel(model=ASR_DIR, **base_model_kw(device))


def load_vad(device: str):
    """单 VAD 实例（流式 cache 状态在调用方会话内，模型本身无状态可共享）"""
    from funasr import AutoModel
    return AutoModel(model=VAD_DIR, **base_model_kw(device))


def load_punc(device: str):
    """单标点实例（无状态，可共享）"""
    from funasr import AutoModel
    return AutoModel(model=PUNC_DIR, **base_model_kw(device))
