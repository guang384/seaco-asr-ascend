"""InstrumentedQueue（FIFO 结果邮箱）测试：顺序、有界、坏消息丢弃、统计。"""
import queue as _q

import pytest

from ipc import InstrumentedQueue


@pytest.fixture()
def q(fakeredis_client):
    return InstrumentedQueue(maxsize=3, index=0, name="gateway_queue_0")


def test_fifo_order(q):
    for i in range(3):
        q.put({"n": i})
    assert [q.get()["n"] for _ in range(3)] == [0, 1, 2]


def test_bounded_put_nowait(q):
    for _ in range(3):
        q.put_nowait({"x": 1})
    with pytest.raises(_q.Full):
        q.put_nowait({"x": 2})
    q.get_nowait()
    q.put_nowait({"x": 3})          # 腾位后可入


def test_get_timeout_raises_empty(q):
    with pytest.raises(_q.Empty):
        q.get(timeout=0.05)


def test_corrupt_message_dropped_not_fatal(q, fakeredis_client):
    fakeredis_client.rpush(q._key, b"not-a-pickle")
    q.put({"ok": 1})
    assert q.get()["ok"] == 1       # 坏消息被跳过，好消息照常交付


def test_stats_shape(q):
    q.put({"a": 1})
    q.get()
    s = q.stats()
    assert s["in"] == 1 and s["out"] == 1 and s["depth"] == 0
    assert "avg_wait_ms" in s
