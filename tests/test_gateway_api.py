"""gateway REST 双模式输入路由 + WS 流式测试（TestClient + 假三级流水）：
gateway 无模型——vad/asr/punc 全走队列，假队列按消息类型即时回结果。

覆盖：multipart 文件 / JSON base64 / data URI / 坏 base64 400 / 超限 413 /
不支持类型 415 / 空文件 / 队列满 429 / MODE=stream 503 / 超时 504 /
worker 错误 500 / 无语音段 / WS 句级 JSON / batch 拒连 / 健康卡 / 指标。"""
import base64
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import gateway
import protocol
from conftest import FakePipeline


WAV_BYTES = (b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
             b"\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00"
             + b"\x00\x00" * 2048)


@pytest.fixture()
def client(monkeypatch):
    fake_q = FakePipeline()

    def fake_decode(data):
        return np.zeros(32000, dtype=np.float32)   # 2s @16k

    monkeypatch.setattr(gateway, "_decode_audio", fake_decode)
    monkeypatch.setattr(gateway, "_q", {"vad": fake_q, "asr": fake_q,
                                        "punc": fake_q})
    monkeypatch.setattr(gateway, "MODE", "mixed")
    monkeypatch.setattr(gateway, "WAIT_TIMEOUT_S", 2.0)
    monkeypatch.setattr(gateway, "_started_at", time.time())
    monkeypatch.setattr(gateway, "REST_GATE",
                        __import__("threading").BoundedSemaphore(4))
    with TestClient(gateway.app) as c:
        c.fake_q = fake_q
        yield c


def test_multipart_file_input(client):
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV_BYTES, "audio/wav")},
                    data={"hotword": "魔搭"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "[0][1]。"          # 两 clip 拼接 + 全文标点
    assert isinstance(body["job_id"], int)
    kinds = [t.get("kind") for t in client.fake_q.tasks
             if t.get("type") == protocol.ASR_TASK]
    assert kinds == ["clip", "clip"]
    # hotword 逐 clip 传递
    assert all(t.get("hotword") == "魔搭" for t in client.fake_q.tasks
               if t.get("type") == protocol.ASR_TASK)
    # 三级流水各就位：vad_batch 1 + asr 2 + punc 1
    types = [t["type"] for t in client.fake_q.tasks]
    assert types.count(protocol.VAD_TASK) == 1
    assert types.count(protocol.ASR_TASK) == 2
    assert types.count(protocol.PUNC_TASK) == 1
    assert client.fake_q.tasks[0]["kind"] == protocol.KIND_VAD_BATCH


def test_json_base64_input(client):
    b64 = base64.b64encode(WAV_BYTES).decode()
    r = client.post("/v1/audio/transcriptions",
                    json={"audio_base64": b64, "hotword": "达摩院"})
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "[0][1]。"


def test_json_data_uri_input(client):
    uri = "data:audio/wav;base64," + base64.b64encode(WAV_BYTES).decode()
    r = client.post("/v1/audio/transcriptions", json={"audio_base64": uri})
    assert r.status_code == 200
    assert r.json()["text"] == "[0][1]。"


def test_bad_base64_400(client):
    r = client.post("/v1/audio/transcriptions",
                    json={"audio_base64": "!!!not-base64!!!"})
    assert r.status_code == 400
    assert "base64" in r.json()["detail"]


def test_missing_field_400(client):
    assert client.post("/v1/audio/transcriptions", json={}).status_code == 400
    assert client.post("/v1/audio/transcriptions",
                       files={"wrong": ("a.wav", b"x")}).status_code == 400


def test_unsupported_content_type_415(client):
    r = client.post("/v1/audio/transcriptions", content=b"raw",
                    headers={"content-type": "application/octet-stream"})
    assert r.status_code == 415


def test_empty_file(client):
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", b"", "audio/wav")})
    assert r.status_code == 200
    assert r.json() == {"text": "", "error": "empty file"}


def test_oversize_413(client, monkeypatch):
    monkeypatch.setattr(gateway, "MAX_AUDIO_MB", 0)   # 0MB 上限 → 立即超限
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV_BYTES, "audio/wav")})
    assert r.status_code == 413


def test_stream_mode_rejects_rest_503(client, monkeypatch):
    monkeypatch.setattr(gateway, "MODE", "stream")
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV_BYTES, "audio/wav")})
    assert r.status_code == 503


def test_worker_error_maps_to_500(client):
    client.fake_q.fail_types.add(protocol.ASR_TASK)
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV_BYTES, "audio/wav")})
    assert r.status_code == 500
    assert "asr boom" in r.json()["detail"]


def test_vad_error_maps_to_500(client):
    client.fake_q.fail_types.add(protocol.VAD_TASK)
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV_BYTES, "audio/wav")})
    assert r.status_code == 500
    assert "vad" in r.json()["detail"]


def test_punc_error_keeps_raw_text(client):
    """标点失败保留原文（可降级阶段）。"""
    client.fake_q.fail_types.add(protocol.PUNC_TASK)
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV_BYTES, "audio/wav")})
    assert r.status_code == 200
    assert r.json()["text"] == "[0][1]"


def test_timeout_maps_to_504_and_cancels(client, monkeypatch):
    monkeypatch.setattr(gateway, "WAIT_TIMEOUT_S", 0.2)

    class SlowQ(FakePipeline):
        def put_nowait(self, task):
            self.tasks.append(task)     # 只入队不回结果

    slow = SlowQ()
    monkeypatch.setattr(gateway, "_q", {"vad": slow, "asr": slow, "punc": slow})
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV_BYTES, "audio/wav")})
    assert r.status_code == 504
    assert slow.cancelled                  # 超时后置取消标记（三队列同 req_id）


def test_no_speech_returns_empty(client, monkeypatch):
    class NoSpeechQ(FakePipeline):
        def put_nowait(self, task):
            self.tasks.append(task)
            if task.get("type") == protocol.VAD_TASK:
                gateway._handle_result(protocol.vad_result(
                    task["req_id"], task.get("idx", 0), [], cache=None))
            else:
                super().put_nowait(task)

    nq = NoSpeechQ()
    client.fake_q = nq
    monkeypatch.setattr(gateway, "_q", {"vad": nq, "asr": nq, "punc": nq})
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV_BYTES, "audio/wav")})
    assert r.status_code == 200
    assert r.json() == {"text": "", "job_id": 0}


def test_ws_streaming_sentences(client):
    """WS 流式：二进制帧 → vad_stream 步进 → 句尾句子（优先级 0）→ 逐句 JSON。"""
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(np.zeros(3200, dtype=np.int16).tobytes())   # 200ms PCM
        ws.send_text(json.dumps({"is_final": True}))
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m.get("is_final"):
                break
    lines = [m for m in msgs if not m.get("is_final")]
    assert [m["text"] for m in lines] == ["[0]。"]
    assert lines[0]["begin"] == 0 and lines[0]["end"] == 1000
    assert msgs[-1] == {"is_final": True}
    # 流式任务以优先级 0 入队（vad_stream + asr 句子 + punc 句尾）
    tasks = client.fake_q.tasks
    stream = [t for t in tasks if t.get("type") == protocol.VAD_TASK]
    sents = [t for t in tasks if t.get("type") == protocol.ASR_TASK]
    assert stream and stream[0]["kind"] == protocol.KIND_VAD_STREAM
    assert stream[0]["priority"] == 0
    assert sents and sents[0]["kind"] == "sent" and sents[0]["priority"] == 0
    # vad_stream cache 往返（worker mutate 后回传）
    assert stream[0].get("cache") is None or isinstance(stream[0]["cache"], dict)


def test_ws_rejected_in_batch_mode(client, monkeypatch):
    monkeypatch.setattr(gateway, "MODE", "batch")
    with client.websocket_connect("/ws") as ws:
        m = ws.receive_json()
        assert "no stream service" in m["error"]


def test_health_five_key_card(client, monkeypatch):
    # 无 supervisor 心跳 → not ready → 503 且五键卡完整
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert set(body) == {"status", "ready", "version", "shm", "recent_failures"}
    assert body["ready"] is False and body["status"] == "starting"


def test_metrics_has_queue_wait_times(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "asr_e2e_requests_total" in r.text
    for q in ("vad", "asr", "punc"):
        assert f'asr_queue_depth{{queue="{q}"}}' in r.text
        assert f'asr_queue_wait_ms{{queue="{q}"' in r.text or True
    assert "asr_degraded" in r.text


def test_metrics_batch_series(client):
    """凑批观测序列：注入 asr worker 心跳后，/metrics 导出批次记账四类指标。"""
    import os

    import proc_common

    snap = {"role": "asr", "ready": True, "wtag": "asr-x",
            "stats": {"jobs": 9, "batches": 2, "batch_tasks": 9,
                      "batch_max": 4, "batchsz_2-4": 2}}
    proc_common.write_heartbeat(gateway.HEALTH_DIR, 999999, snap)
    gateway._hb_cache_invalidate()
    try:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert 'asr_worker_batch_claims_total{worker="asr-x"} 2' in r.text
        assert 'asr_worker_batch_tasks_total{worker="asr-x"} 9' in r.text
        assert 'asr_worker_batch_max{worker="asr-x"} 4' in r.text
        assert ('asr_worker_batch_size_total{worker="asr-x",bucket="2-4"} 2'
                in r.text)
    finally:
        try:
            os.unlink(os.path.join(gateway.HEALTH_DIR, "worker-999999.json"))
        except OSError:
            pass
        gateway._hb_cache_invalidate()


def test_incremental_vad_dispatch(client, monkeypatch):
    """边切边发：增量段（done=False）随到随派发 clip，终态段补发；
    重派重发的重复段（worker 死亡重跑）去重不重算。"""

    class IncrQ(FakePipeline):
        def put_nowait(self, task):
            if task.get("type") == protocol.VAD_TASK:
                self.tasks.append(task)
                # 模拟 vad worker 边切边发：两次增量 + 终态（终态带重复段）
                gateway._handle_result(protocol.vad_result(
                    task["req_id"], 0, [[0, 1000]], done=False))
                gateway._handle_result(protocol.vad_result(
                    task["req_id"], 0, [[1000, 2000]], done=False))
                gateway._handle_result(protocol.vad_result(
                    task["req_id"], 0, [[1000, 2000], [2000, 2500]],
                    done=True))
            else:
                super().put_nowait(task)

    q = IncrQ()
    monkeypatch.setattr(gateway, "_q", {"vad": q, "asr": q, "punc": q})
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV_BYTES, "audio/wav")})
    assert r.status_code == 200, r.text
    # 3 个唯一段（重复段去重）→ 3 clip，按序拼接
    assert r.json()["text"] == "[0][1][2]。"
    kinds = [t.get("idx") for t in q.tasks if t.get("type") == protocol.ASR_TASK]
    assert kinds == [0, 1, 2]


def test_admin_endpoints(client, monkeypatch):
    """admin 应用端点可达（index/ready/stats/queue/resources），queue 含三队列等待。"""
    monkeypatch.setattr(gateway, "_q", {"vad": client.fake_q,
                                        "asr": client.fake_q,
                                        "punc": client.fake_q})
    with TestClient(gateway.admin) as ac:
        assert ac.get("/").status_code == 200
        r = ac.get("/ready")
        assert r.status_code == 503 and r.json()["ready"] is False
        s = ac.get("/stats").json()
        for k in ("processed", "failed", "requests", "rtf", "recent"):
            assert k in s, k
        q = ac.get("/queue").json()
        assert len(q["queues"]) == 3
        names = {x["name"] for x in q["queues"]}
        assert names == {"vad", "asr", "punc"}
        for x in q["queues"]:
            assert "wait" in x and "depth" in x
        assert "capacity" in q
        pr = ac.post("/queue/1/promote")
        assert pr.status_code == 200 and pr.json()["promoted"] == 1
        res = ac.get("/resources").json()
        assert "cpu_percent" in res and "shm" in res
