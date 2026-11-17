"""protocol 消息契约测试：构造函数字段完整性 + put_result 丢弃计数。"""
import queue as _q

import protocol


def test_asr_task_fields():
    t = protocol.asr_task("5", 2, 1, protocol.KIND_CLIP,
                          {"shm_name": "asr_x", "offset": 100, "samples": 320},
                          hotword="魔搭", priority=100)
    for k in ("type", "req_id", "idx", "origin", "kind", "piece",
              "hotword", "priority"):
        assert k in t, k
    assert t["type"] == protocol.ASR_TASK
    assert t["req_id"] == "5" and t["idx"] == 2 and t["origin"] == 1
    assert t["priority"] == 100
    assert t["piece"]["samples"] == 320


def test_asr_result_fields():
    r = protocol.asr_result("5", 2, protocol.KIND_SENT, text="你好")
    assert r["type"] == protocol.ASR_RESULT
    assert r["text"] == "你好" and r["error"] is None
    e = protocol.asr_result("5", 2, error="boom")
    assert e["error"] == "boom" and e["text"] is None


def test_put_result_success_and_drop_count():
    class _Q:
        def put(self, msg, timeout=None):
            if msg.get("fail"):
                raise _q.Full()
    qq = _Q()
    before = protocol.drop_count()
    assert protocol.put_result(qq, {"req_id": "1"}) is True
    assert protocol.put_result(qq, {"req_id": "2", "fail": True}) is False
    assert protocol.drop_count() == before + 1


def test_kinds():
    assert protocol.KIND_SENT == "sent" and protocol.KIND_CLIP == "clip"
    assert protocol.RESULT_PUT_TIMEOUT_S > 0


def test_vad_task_and_result():
    t = protocol.vad_task("5", 0, 1, protocol.KIND_VAD_STREAM,
                          chunk="ndarray-placeholder", cache={"k": 1},
                          is_final=True, priority=0)
    for k in ("type", "req_id", "idx", "origin", "kind", "priority",
              "is_final", "chunk_size", "max_end_silence_time", "chunk", "cache"):
        assert k in t, k
    assert t["type"] == protocol.VAD_TASK and t["kind"] == "vad_stream"
    b = protocol.vad_task("5", 0, 1, protocol.KIND_VAD_BATCH,
                          piece={"shm_name": "x", "offset": 0, "samples": 9})
    assert "piece" in b and "chunk" not in b
    r = protocol.vad_result("5", 0, [[0, 100], [-1, -1]], cache={"k": 2})
    assert r["type"] == protocol.VAD_RESULT
    assert r["segments"] == [[0, 100], [-1, -1]] and r["cache"] == {"k": 2}
    e = protocol.vad_result("5", 0, [], error="boom")
    assert e["error"] == "boom"


def test_punc_task_and_result():
    t = protocol.punc_task("5", 0, 1, "你好", priority=0)
    assert t["type"] == protocol.PUNC_TASK and t["text"] == "你好"
    r = protocol.punc_result("5", 0, "你好。")
    assert r["type"] == protocol.PUNC_RESULT and r["text"] == "你好。"
    e = protocol.punc_result("5", 0, "你好", error="x")
    assert e["error"] == "x"


def test_wait_sample_stats(fakeredis_client):
    import ipc
    assert ipc.wait_stats("asr")["n"] == 0
    for v in (10.0, 20.0, 30.0, 40.0, 100.0):
        ipc.push_wait_sample("asr", v)
    w = ipc.wait_stats("asr")
    assert w["n"] == 5 and w["avg_ms"] == 40.0
    assert w["p50_ms"] == 30.0 and w["max_ms"] == 100.0 and w["p95_ms"] == 100.0
    # 滚动窗口 100 条封顶
    for i in range(120):
        ipc.push_wait_sample("vad", float(i))
    assert ipc.wait_stats("vad")["n"] == 100
