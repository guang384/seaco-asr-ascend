"""心跳/看门狗判决测试：原子落盘、stale/stuck 判定。"""
import json
import os
import time

import proc_common
import supervisor


def test_write_heartbeat_atomic(tmp_path):
    proc_common.write_heartbeat(str(tmp_path), 123, {"role": "asr", "ready": True})
    path = tmp_path / "worker-123.json"
    assert path.exists()
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["role"] == "asr" and d["pid"] == 123 and "ts" in d


def test_classify_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "HEARTBEAT_STALE_S", 15.0)
    monkeypatch.setattr(supervisor, "STUCK_KILL_S", 300.0)
    p = tmp_path / "worker-1.json"
    p.write_text(json.dumps({"stuck_s": 0}), encoding="utf-8")
    old = time.time() - 60
    os.utime(p, (old, old))
    verdict, detail = supervisor.classify_heartbeat(str(p), time.time())
    assert verdict == "stale"


def test_classify_stuck(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "HEARTBEAT_STALE_S", 15.0)
    monkeypatch.setattr(supervisor, "STUCK_KILL_S", 300.0)
    p = tmp_path / "worker-2.json"
    p.write_text(json.dumps({"stuck_s": 400.0}), encoding="utf-8")
    verdict, detail = supervisor.classify_heartbeat(str(p), time.time())
    assert verdict == "stuck" and "inference hang" in detail


def test_classify_ok_and_tolerant(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "HEARTBEAT_STALE_S", 15.0)
    monkeypatch.setattr(supervisor, "STUCK_KILL_S", 300.0)
    p = tmp_path / "worker-3.json"
    p.write_text(json.dumps({"stuck_s": 1.0}), encoding="utf-8")
    assert supervisor.classify_heartbeat(str(p), time.time())[0] == "ok"
    # 坏快照：宽容按 ok（真死走 stale 判定）
    p2 = tmp_path / "worker-4.json"
    p2.write_text("{half json", encoding="utf-8")
    assert supervisor.classify_heartbeat(str(p2), time.time())[0] == "ok"
    # 缺失文件：宽容
    assert supervisor.classify_heartbeat(str(tmp_path / "nope.json"),
                                         time.time())[0] == "ok"
