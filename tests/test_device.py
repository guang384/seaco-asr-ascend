"""asr_common.resolve_device 设备语义矩阵：
- auto（默认）：NPU→XPU→CUDA→CPU 逐级降级**永不退出**（回归锁：20261118 语义修正）
- 显式 DEVICE=npu：探卡失败 SystemExit fail-fast

**绝不 import 真 torch**（其 aarch64 C 扩展在 QEMU 仿真下段错误，真实事故
20261118 两连）——torch/torch_npu 全部经 sys.modules 注入假件，本地与镜像
行为一致。"""
import sys
import types

import pytest

import asr_common


def _fake_torch(monkeypatch, *, torch_npu=True, npu_present=True,
                npu_available=True, device_count=1, xpu=False, cuda=False):
    """向 sys.modules 注入假 torch + torch_npu——resolve_device /
    pick_npu_index 内的 import 命中假件，绝无真 torch 加载。
    torch_npu=False → sys.modules 放 None，import 必抛 ImportError。"""
    if torch_npu:
        monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    else:
        monkeypatch.setitem(sys.modules, "torch_npu", None)
    npu = None
    if npu_present:
        npu = types.SimpleNamespace(
            is_available=lambda: npu_available,
            device_count=lambda: device_count,
            synchronize=lambda dev: None,
        )
    fake = types.SimpleNamespace(
        npu=npu,
        xpu=types.SimpleNamespace(is_available=lambda: xpu),
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)


def _probe_fail(msg="aicore hang"):
    return lambda idx=0: (_ for _ in ()).throw(RuntimeError(msg))


def _no_probe():
    return lambda idx=0: None


# ── auto：逐级降级，永不退出 ──

def test_auto_probe_fail_falls_to_cpu(monkeypatch, caplog):
    """回归锁：auto 档 NPU 探卡失败必须降级继续，绝不 SystemExit。"""
    _fake_torch(monkeypatch, npu_available=True)
    monkeypatch.setattr(asr_common, "_npu_probe", _probe_fail())
    assert asr_common.resolve_device("auto") == "cpu"
    assert any("降级继续" in r.message for r in caplog.records)


def test_auto_probe_ok_returns_npu(monkeypatch):
    _fake_torch(monkeypatch, npu_available=True)
    monkeypatch.setattr(asr_common, "_npu_probe", _no_probe())
    assert asr_common.resolve_device("auto") == "npu:0"


def test_auto_npu_unavailable_falls_to_xpu(monkeypatch):
    _fake_torch(monkeypatch, npu_available=False, xpu=True)
    monkeypatch.setattr(asr_common, "_npu_probe", _no_probe())
    assert asr_common.resolve_device("auto") == "xpu"


def test_auto_torch_npu_missing_falls_to_cpu(monkeypatch):
    _fake_torch(monkeypatch, torch_npu=False)              # import 必抛 ImportError
    assert asr_common.resolve_device("auto") == "cpu"


def test_auto_probe_fail_chain_continues_to_xpu(monkeypatch):
    """NPU 探卡失败后降级链不中断：XPU 档照常尝试（且在 CUDA 之前）。"""
    _fake_torch(monkeypatch, npu_available=True, xpu=True, cuda=True)
    monkeypatch.setattr(asr_common, "_npu_probe", _probe_fail())
    assert asr_common.resolve_device("auto") == "xpu"


def test_auto_cuda_fallback_when_xpu_off(monkeypatch):
    _fake_torch(monkeypatch, npu_available=False, cuda=True)
    monkeypatch.setattr(asr_common, "_npu_probe", _no_probe())
    assert asr_common.resolve_device("auto") == "cuda"


# ── 显式 DEVICE=npu：fail-fast ──

def test_explicit_npu_import_fail_exits(monkeypatch):
    _fake_torch(monkeypatch, torch_npu=False)
    with pytest.raises(SystemExit):
        asr_common.resolve_device("npu")


def test_explicit_npu_no_card_exits(monkeypatch):
    _fake_torch(monkeypatch, npu_available=False)
    with pytest.raises(SystemExit):
        asr_common.resolve_device("npu")


def test_explicit_npu_probe_fail_exits(monkeypatch):
    _fake_torch(monkeypatch, npu_available=True)
    monkeypatch.setattr(asr_common, "_npu_probe", _probe_fail("bad card"))
    with pytest.raises(SystemExit):
        asr_common.resolve_device("npu")


def test_explicit_npu_ok(monkeypatch):
    _fake_torch(monkeypatch, npu_available=True)
    monkeypatch.setattr(asr_common, "_npu_probe", _no_probe())
    assert asr_common.resolve_device("npu") == "npu:0"


# ── 显式 cpu / cuda / xpu 透传 ──

def test_explicit_values_passthrough(monkeypatch):
    assert asr_common.resolve_device("cpu") == "cpu"
    _fake_torch(monkeypatch)
    assert asr_common.resolve_device("cuda") == "cuda"
    assert asr_common.resolve_device("xpu") == "xpu"


# ── pick_npu_index 卡号发现 ──

def test_pick_npu_index_explicit_env_wins(monkeypatch):
    _fake_torch(monkeypatch, device_count=4)
    monkeypatch.setenv("ASCEND_DEVICE_ID", "3")
    assert asr_common.pick_npu_index() == 3   # 显式指定优先，不因越界丢弃


def test_pick_npu_index_pid_spread(monkeypatch):
    _fake_torch(monkeypatch, device_count=4)
    monkeypatch.delenv("ASCEND_DEVICE_ID", raising=False)
    import os
    assert asr_common.pick_npu_index() == os.getpid() % 4   # 枚举打散


def test_pick_npu_index_npu_smi_fallback(monkeypatch):
    """torch 枚举 0 卡 → npu-smi 驱动层兜底（解析 chip 列，pid 取模打散）。"""
    _fake_torch(monkeypatch, device_count=0)
    monkeypatch.delenv("ASCEND_DEVICE_ID", raising=False)
    out = ("Card  Chip   Name    Health\n"
           "  0   5    910B3   OK\n"
           "  1   6    910B3   OK")
    monkeypatch.setattr(asr_common.subprocess, "run",
                        lambda *a, **kw: types.SimpleNamespace(stdout=out))
    assert asr_common.pick_npu_index() in (5, 6)
