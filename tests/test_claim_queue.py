"""AsrClaimQueue 语义测试（fakeredis + Lua）：优先级序、认领/ack、
worker 死亡重派（重试一次再报错）、毒块弃置、插队、取消标记。"""
import time

import pytest

import protocol
from ipc import (PRI_CLIP, PRI_PROMOTED, PRI_SENT, AsrClaimQueue, task_score)


@pytest.fixture()
def q(fakeredis_client):
    return AsrClaimQueue(maxsize=100, index=0, name="asr")


def _task(req_id="1", idx=0, priority=PRI_CLIP, samples=10, **kw):
    # 走 protocol 构造函数（防字段漂移：任务消息字段以 protocol 为单一事实来源）
    t = protocol.asr_task(str(req_id), idx, 0, protocol.KIND_CLIP,
                          {"shm_name": "x", "offset": 0, "samples": samples},
                          priority=priority)
    t.update(kw)
    return t


def test_priority_order_sentence_first(q):
    """统一池语义：句子（0）先于 clip（100/10）被认领，同优先级 FIFO。"""
    q.put(_task(req_id="clip1", priority=PRI_CLIP))
    q.put(_task(req_id="sent1", priority=PRI_SENT))
    q.put(_task(req_id="clip2", priority=PRI_PROMOTED))
    q.put(_task(req_id="sent2", priority=PRI_SENT))
    got = [c["req_id"] for _m, c in [q.claim(1, "w-1")[0] for _ in range(4)]]
    assert got == ["sent1", "sent2", "clip2", "clip1"]


def test_claim_moves_to_inflight_and_ack_clears(q):
    q.put(_task())
    claimed = q.claim(1, "asr-0-123")
    assert len(claimed) == 1
    member, chunk = claimed[0]
    assert q.depth() == 0                       # 已出主队列
    infl = q._redis().zrange("asr:q:0:asr:infl:asr-0-123", 0, -1)
    assert len(infl) == 1                       # 在途账本
    assert q.ack([member], "asr-0-123") == 1
    assert q._redis().zcard("asr:q:0:asr:infl:asr-0-123") == 0


def test_requeue_worker_retries_once_then_gives_up(fakeredis_client):
    """用户决策"重试一次再报错"：首次死亡重派（attempts=1≤1），
    二次死亡弃置并走 on_giveup 回调（客户端立即 500）。"""
    q = AsrClaimQueue(maxsize=100, index=0, name="asr")
    q.put(_task(req_id="7"))
    member, _ = q.claim(1, "asr-0-111")[0]
    # 第一次死亡：重派回主队列
    n = q.requeue_worker("asr-0-111", max_redispatch=1)
    assert n == 1
    assert q.depth() == 1
    # 第二次被认领后死亡：超限弃置 + on_giveup 收口
    member2, _ = q.claim(1, "asr-1-222")[0]
    given = []
    n = q.requeue_worker("asr-1-222", max_redispatch=1,
                         on_giveup=lambda chunk: given.append(chunk))
    assert n == 0
    assert q.depth() == 0
    assert len(given) == 1 and given[0]["req_id"] == "7"
    assert q.stats()["bad"] == 1


def test_requeue_preserves_priority_and_order(fakeredis_client):
    """重派 score 按原优先级+原 FIFO 序号复算：句子仍先于 clip 出队。"""
    q = AsrClaimQueue(maxsize=100, index=0, name="asr")
    q.put(_task(req_id="c", priority=PRI_CLIP))
    q.put(_task(req_id="s", priority=PRI_SENT))
    claimed = [q.claim(1, f"w-{i}")[0] for i in range(2)]   # 两条都进在途
    for i, (m, _c) in enumerate(claimed):
        q.requeue_worker(f"w-{i}", max_redispatch=None)
    order = [q.claim(1, "w2")[0][1]["req_id"] for _ in range(2)]
    assert order == ["s", "c"]


def test_sweep_dead_worker_full_requeue_alive_only_ttl(fakeredis_client):
    """sweep：死者在途立即全量重派；活者仅超长 TTL 兜底（慢 worker 不误伤）。"""
    q = AsrClaimQueue(maxsize=100, index=0, name="asr")
    q.put(_task(req_id="d"))       # 死者持有
    q.put(_task(req_id="a"))       # 活者持有
    q.claim(1, "asr-0-999")        # pid 999（死）
    q.claim(1, "asr-1-1000")       # pid 1000（活）
    r = q.sweep(alive_pids={1000}, now=time.time(), ttl=600,
                max_redispatch=1)
    assert r["redispatched"] == 1 and r["gaveup"] == 0
    ids = [c["req_id"] for _m, c in [q.claim(1, "w")[0] for _ in range(1)]]
    assert ids == ["d"]                       # 死者任务回来了
    assert q.depth() == 0                     # 活者的还没动


def test_sweep_giveup_over_limit_notifies(fakeredis_client):
    """sweep 毒块：重派超限 → 弃置 + on_giveup。"""
    q = AsrClaimQueue(maxsize=100, index=0, name="asr")
    q.put(_task(req_id="p"))
    q.claim(1, "asr-0-999")
    q.requeue_worker("asr-0-999", max_redispatch=1)   # 第一次重派（计数 1）
    q.claim(1, "asr-0-998")
    given = []
    r = q.sweep(alive_pids=set(), now=time.time(), ttl=600,
                max_redispatch=1, on_giveup=lambda c: given.append(c))
    assert r["gaveup"] == 1 and r["redispatched"] == 0
    assert given and given[0]["req_id"] == "p"


def test_bad_member_dropped_not_fatal(q, fakeredis_client):
    """坏 member（截断/外部误写）被丢弃计数，claim 不炸；
    坏 member 成为队首时独占其类（bucket 来自垃圾字节），下一轮正常任务照常出。"""
    fakeredis_client.zadd("asr:q:0:asr", {b"garbage-no-seq-header": 1})
    q.put(_task(req_id="ok"))
    first = q.claim(2, "w-1")      # 队首是坏 member：本批只碰到它（丢弃计数）
    second = q.claim(2, "w-1")     # 下一轮正常任务照常出
    ids = [c["req_id"] for _m, c in first + second]
    assert ids == ["ok"]
    assert q.stats()["bad"] == 1


def test_promote_rescores_only_that_req(q):
    q.put(_task(req_id="1"))
    q.put(_task(req_id="2"))
    assert q.promote("1", PRI_PROMOTED) == 1
    got = [c["req_id"] for _m, c in [q.claim(1, "w")[0] for _ in range(2)]]
    assert got == ["1", "2"]        # 1 插队到 2 前面
    assert q.promote("nope") == 0


def test_cancel_marking(q, fakeredis_client):
    q.cancel_add("req-9")
    assert q.is_cancelled("req-9")
    assert not q.is_cancelled("req-8")
    assert q.cancelled_set(["req-9", "req-8"]) == {"req-9"}


def test_put_nowait_full_raises(q):
    small = AsrClaimQueue(maxsize=1, index=0, name="asr")
    small.put_nowait(_task())
    import queue as _q
    with pytest.raises(_q.Full):
        small.put_nowait(_task())


def test_score_formula():
    assert task_score(PRI_SENT, 5) < task_score(PRI_PROMOTED, 1)
    assert task_score(PRI_PROMOTED, 5) < task_score(PRI_CLIP, 1)


def test_claim_same_bucket_batch(q):
    """同长度桶凑批：claim(N) 只取与队首同桶（±2s）的任务——
    "队列里有和当前要处理的差不多长度的，那就取出来"。"""
    q.put(_task(req_id="a", samples=32000 * 2))         # 4.0s → 桶 2
    q.put(_task(req_id="b", samples=32000 * 2 + 100))   # ~4.0s → 桶 2
    q.put(_task(req_id="c", samples=32000 * 5))         # 10.0s → 桶 5
    q.put(_task(req_id="d", samples=32000 * 2 + 200))   # ~4.0s → 桶 2
    got = [c["req_id"] for _m, c in q.claim(3, "w-1")]
    assert sorted(got) == ["a", "b", "d"]     # 同桶三条一批拿走
    assert [c["req_id"] for _m, c in q.claim(3, "w-1")] == ["c"]   # 异桶下一轮
