"""compute 批前向测试（stub 模型）：同桶凑批一次 generate、hotword 分组、
OOM 重试一次后降级逐条、坏块单条隔离。"""
import numpy as np
import pytest

import compute
import protocol


class StubModel:
    """记录调用形状的假模型：list 输入走批（回 "b{i}"），单条回 "s"。"""

    def __init__(self):
        self.calls = []
        self.batch_fail = 0      # 前 N 次批调用抛 OOM

    def generate(self, input=None, disable_pbar=True, batch_size=None,
                 hotword=None):
        self.calls.append(("list" if isinstance(input, list) else "single",
                           batch_size, hotword))
        if isinstance(input, list):
            if self.batch_fail > 0:
                self.batch_fail -= 1
                raise RuntimeError("out of memory")
            return [{"text": f"b{i}"} for i in range(len(input))]
        return [{"text": "s"}]


class FakeQ:
    def __init__(self):
        self.msgs = []

    def put(self, msg, timeout=None):
        self.msgs.append(msg)


class FakeClaim:
    def is_cancelled(self, rid):
        return False


def _mk(idx, samples=32000, hw=None):
    # 走 protocol 构造函数（防字段漂移：任务消息字段以 protocol 为单一事实来源）
    return protocol.asr_task("9", idx, 0, protocol.KIND_CLIP,
                             {"shm_name": "x", "offset": 0,
                              "samples": samples},
                             hotword=hw, priority=100)


@pytest.fixture()
def env(monkeypatch):
    stub = StubModel()
    monkeypatch.setitem(compute._state, "model", stub)
    monkeypatch.setitem(compute._state, "device", "cpu")
    monkeypatch.setattr(compute, "_read_task_audio",
                        lambda t: np.zeros(t["piece"]["samples"],
                                           dtype=np.float32))
    monkeypatch.setattr(compute, "_push_capacity_sample", lambda *a: None)
    fq = FakeQ()
    return stub, fq


def test_batch_forward_one_generate(env):
    stub, fq = env
    b0 = compute._stats["batches"]
    t0 = compute._stats["batch_tasks"]
    s0 = compute._stats.get("batchsz_2-4", 0)
    tasks = [_mk(0), _mk(1)]
    compute._run_asr_jobs([("m0", tasks[0]), ("m1", tasks[1])], [fq], FakeClaim())
    # 一次 list 批前向（batch_size=2），逐条回投 b0/b1
    assert stub.calls == [("list", 2, None)]
    texts = {m["idx"]: m["text"] for m in fq.msgs if m["type"] == "asr_result"}
    assert texts == {0: "b0", 1: "b1"}
    # 批次记账：凑批 1 次、接纳 2 条、落 2-4 档
    assert compute._stats["batches"] == b0 + 1
    assert compute._stats["batch_tasks"] == t0 + 2
    assert compute._stats["batchsz_2-4"] == s0 + 1


def test_hotword_groups_split(env):
    stub, fq = env
    tasks = [_mk(0, hw="魔搭"), _mk(1, hw="达摩"), _mk(2, hw="魔搭")]
    compute._run_asr_jobs([("m0", tasks[0]), ("m1", tasks[1]),
                           ("m2", tasks[2])], [fq], FakeClaim())
    # 两个 hotword 桶 → 两次批调用（2 条 + 1 条）
    lists = [c for c in stub.calls if c[0] == "list"]
    assert sorted(c[1] for c in lists) == [1, 2]
    assert {c[2] for c in lists} == {"魔搭", "达摩"}


def test_oom_retries_then_fallback_sequential(env):
    stub, fq = env
    before = compute._stats["oom_retries"]
    stub.batch_fail = 1                # 首次批前向 OOM → 重试整批成功
    tasks = [_mk(0), _mk(1)]
    compute._run_asr_jobs([("m0", tasks[0]), ("m1", tasks[1])], [fq], FakeClaim())
    assert [c[0] for c in stub.calls] == ["list", "list"]   # 重试仍批
    assert compute._stats["oom_retries"] == before + 1
    assert len(fq.msgs) == 2


def test_bad_block_isolated(env, monkeypatch):
    """坏 SHM 块只伤自家：一条读失败 → 单条 error，其余照常批前向。"""
    stub, fq = env

    def read(t):
        if t["idx"] == 1:
            raise FileNotFoundError("shm gone")
        return np.zeros(t["piece"]["samples"], dtype=np.float32)

    monkeypatch.setattr(compute, "_read_task_audio", read)
    tasks = [_mk(0), _mk(1), _mk(2)]
    compute._run_asr_jobs([("m0", tasks[0]), ("m1", tasks[1]),
                           ("m2", tasks[2])], [fq], FakeClaim())
    errs = [m for m in fq.msgs if m.get("error")]
    oks = [m for m in fq.msgs if m.get("text") is not None and not m.get("error")]
    assert len(errs) == 1 and errs[0]["idx"] == 1
    assert {m["idx"] for m in oks} == {0, 2}
    assert stub.calls == [("list", 2, None)]    # 剩余两条仍一次批前向
