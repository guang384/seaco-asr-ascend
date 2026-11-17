"""pytest 根配置：把项目根目录放进 sys.path（各测试直接 import 顶层模块），
并在任何项目模块导入前固定健康目录到临时位置（心跳文件不落 /tmp 全局）。"""
import atexit
import os
import shutil
import sys
import tempfile
import time

import fakeredis
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_TMP_HEALTH_ROOT = tempfile.mkdtemp(prefix="asr-test-health-")
os.environ.setdefault(
    "ASR_HEALTH_DIR", os.path.join(_TMP_HEALTH_ROOT, "health"))
atexit.register(shutil.rmtree, _TMP_HEALTH_ROOT, ignore_errors=True)

import gateway  # noqa: E402
import ipc  # noqa: E402
import proc_common  # noqa: E402
import protocol  # noqa: E402


@pytest.fixture(autouse=True)
def fakeredis_client(monkeypatch):
    """redis 客户端全局兜底：每测试独立 FakeServer，ipc._make_redis_client 指向
    它——任何测试路径的 redis 操作都在假实例上执行（Windows 无 AF_UNIX；
    真 redis-server 语义由镜像内测试覆盖）。需要真实 redis 的测试在用例内
    重新覆盖 _make_redis_client。"""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeStrictRedis(server=server, decode_responses=False)
    monkeypatch.setattr(ipc, "_make_redis_client", lambda: client)
    monkeypatch.setattr(ipc, "_SHARED_REDIS", None)  # 单例重置
    return client


class FakePipeline:
    """假三级流水队列：入队即按消息类型回结果（模拟 worker 秒回）。
    vad_batch → 两段切分；vad_stream → is_final 出一句（cache 原样回传）；
    asr → "[idx]"；punc → 文本+"。"。可注入失败/延迟。
    放 conftest：gateway REST/WS 语义测试共用，派生子类注入特定行为。"""

    def __init__(self):
        self.tasks = []
        self.fail_types = set()          # 这些消息类型回 error
        self.delay_s = 0.0
        self.cancelled = set()
        self.promoted = []

    def put_nowait(self, task):
        self.tasks.append(task)
        if self.delay_s:
            time.sleep(self.delay_s)
        t = task.get("type")
        req, idx = task["req_id"], task.get("idx", 0)
        if t in self.fail_types:
            msg = {"error": f"{t} boom"}
            if t == protocol.VAD_TASK:
                msg = protocol.vad_result(req, idx, [], error="vad boom")
            elif t == protocol.ASR_TASK:
                msg = protocol.asr_result(req, idx,
                                          task.get("kind", "clip"),
                                          error="asr boom")
            else:
                msg = protocol.punc_result(req, idx, None,
                                           error="punc boom")
        elif t == protocol.VAD_TASK:
            if task.get("kind") == protocol.KIND_VAD_STREAM:
                segs = [[0, 1000]] if task.get("is_final") else []
                msg = protocol.vad_result(req, idx, segs,
                                          cache={"echo": True})
            else:
                msg = protocol.vad_result(req, idx,
                                          [[0, 1000], [1000, 2000]],
                                          cache=None)
        elif t == protocol.ASR_TASK:
            msg = protocol.asr_result(req, idx,
                                      task.get("kind", "clip"),
                                      text=f"[{idx}]")
        else:
            msg = protocol.punc_result(req, idx,
                                       (task.get("text") or "") + "。")
        gateway._handle_result(msg)

    def depth(self):
        return len(self.tasks)

    def cancel_add(self, rid):
        self.cancelled.add(rid)

    def is_cancelled(self, rid):
        return rid in self.cancelled

    def promote(self, rid, pri=10):
        self.promoted.append((rid, pri))
        return 1

    def stats(self):
        return {"depth": 0, "in": 0, "out": 0, "bad": 0, "avg_wait_ms": 0}

    def claim(self, *a, **kw):
        return []
