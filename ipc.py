"""跨进程 Redis 传输层：结果邮箱（FIFO）+ 认领式任务队列 + 客户端/托管。

被 gateway（准入/观测/入队）/ compute（认领消费）/ supervisor（编排守护）
复用；机制移植自 kokoro-tts-ascend 生产验证口径（mp.Queue 信号量被 SIGKILL
永久锁死、ZeroMQ 队头阻塞两段事故史之后落在 Redis：消息常驻服务端、LPOP
拉取、计数服务端单线程原子）。

与 kokoro 版的差异（ASR 口径）：
- 任务队列为优先级感知认领式（ZSET score = 优先级类 × 10^13 + FIFO 序号）：
  WS 句子（0）永远先于 REST clip（100，promote 后 10）被认领——原统一池
  heapq 语义的跨进程等价物。
- member 头 = 2B 时长桶 + 20B FIFO 序号 + pickle 载荷：时长桶（≈2s @16k）
  供认领 Lua 取"与队首差不多长度"的同桶凑批（ASR 批前向 padding 浪费有界）。
- 重派给 up callback：给毒块/超限放弃的 chunk 补一条错误结果回 origin 网关
  ——客户端立即 500（带原因），而不是等 WAIT_TIMEOUT 504。
"""
import logging
import os
import pickle
import queue
import struct
import time

import redis

# ── 优先级（统一池语义：数值小者先）──
PRI_SENT = 0        # WS 流式句尾句子
PRI_PROMOTED = 10   # 插队后的 REST clip
PRI_CLIP = 100      # REST 拆段 clip
PRIORITY_BASE = 10 ** 13   # score = priority × PRIORITY_BASE + seq（seq 全局自增）

# ── 传输层参数 ──
IPC_REDIS_SOCKET = os.getenv("IPC_REDIS_SOCKET", "/tmp/asr-redis.sock")
# 外部 redis 模式（IPC_REDIS_URL=tcp://host:port）：不拉起 server、直连
# ——dev/Windows 验证与多副本共享 redis 用；生产留空走 unix socket 自拉
IPC_REDIS_URL = (os.getenv("IPC_REDIS_URL") or "").strip()
_CLAIM_SCAN_WINDOW = 64   # 认领扫描窗口（最老 64 条）
_CLAIM_POLL_S = 0.01      # 容量门等待与空认领轮询间隔
_REDIS_POLL_S = 0.01      # 出队轮询间隔（消息交接延迟 ≤10ms，量级无感）
_MEMBER_HEADER = 22       # member 头：2B 时长桶 + 20B FIFO 序号（十进制 ASCII）
_BUCKET_SAMPLES = 32000   # 时长桶宽：2s @16k（"差不多长度"= 同桶 ±2s，凑批 padding 浪费有界）
_EMA_ALPHA = 0.2          # 队列等待时间 EMA 平滑系数
_WAIT_LIST_MAX = 99       # 等待样本滚动窗口（LPUSH + LTRIM 0..99 = 最近 100 条）

_qlog = logging.getLogger("asr.queue")

_SHARED_REDIS = None   # 进程级共享客户端（cancel SET 等无队列归属的操作用）


def task_score(priority: int, seq: int) -> float:
    """优先级感知 score：数值小者先认领（句子 0 < 插队 10 < clip 100），
    同优先级按 FIFO 序号。重派/插队复算 score 用同一函数（单一公式）。"""
    return priority * PRIORITY_BASE + seq


def _make_redis_client():
    """redis 客户端工厂（每队列每进程一个实例；连接池线程安全）。
    测试经 monkeypatch 注入 fakeredis 实现单进程语义验证。"""
    # socket 超时必须有界：redis-server 活着但 socket 挂起时无超时的命令会
    # 永久阻塞 worker 主循环——表现为"心跳还在、请求全挂"，比快速失败难查。
    if IPC_REDIS_URL:
        url = IPC_REDIS_URL
        if url.startswith("tcp://"):
            url = "redis://" + url[len("tcp://"):]   # 容错：运维惯写 tcp://
        return redis.Redis.from_url(url, decode_responses=False,
                                    socket_timeout=5.0,
                                    socket_connect_timeout=5.0)
    return redis.Redis(unix_socket_path=IPC_REDIS_SOCKET, decode_responses=False,
                       socket_timeout=5.0, socket_connect_timeout=5.0)


def shared_redis():
    """进程级共享 redis 客户端（懒建；无队列归属的轻量操作用）。"""
    global _SHARED_REDIS
    if _SHARED_REDIS is None:
        _SHARED_REDIS = _make_redis_client()
    return _SHARED_REDIS


def cancel_add(req_id: str) -> None:
    """取消标记（gateway 超时/断连路径调用，ZSET score=标记时刻）：
    worker 认领后、推理前检查跳过。标记寿命 600s 由 sweep 周期清扫。"""
    shared_redis().zadd("asr:cancel", {req_id: time.time()})


def is_cancelled(req_id: str) -> bool:
    return shared_redis().zscore("asr:cancel", req_id) is not None


def start_redis_server():
    """拉起 redis-server（unix socket、纯内存、有界）并等就绪。

    supervisor 启动期调用，fail-fast：10s 内 ping 不通直接抛 RuntimeError
    （绝不出现"服务起来了、队列却是黑洞"的空心态）。
    IPC_REDIS_URL 设置时走外部 redis：不拉起 server（返回 None），仅 ping。"""
    if IPC_REDIS_URL:
        _wait_redis_ready(_make_redis_client(),
                          what=f"external redis url={IPC_REDIS_URL}")
        return None
    proc = None
    try:
        proc = __import__("subprocess").Popen(
            ["redis-server", "--unixsocket", IPC_REDIS_SOCKET,
             "--unixsocketperm", "700", "--port", "0",
             "--save", "", "--appendonly", "no",
             "--maxmemory", "256mb", "--maxmemory-policy", "noeviction"],
            stdout=__import__("subprocess").DEVNULL,
            stderr=__import__("subprocess").DEVNULL)
    except FileNotFoundError:
        raise RuntimeError(
            "redis-server 二进制不存在：容器内已由 apt 内置；裸机部署需安装 "
            "redis-server（或设 IPC_REDIS_URL 走外部 redis）") from None
    try:
        _wait_redis_ready(_make_redis_client(),
                          what=f"redis-server socket={IPC_REDIS_SOCKET}")
    except RuntimeError:
        proc.kill()
        raise
    return proc


def _wait_redis_ready(client, what: str, deadline_s: float = 10.0) -> None:
    """10s ping 等就绪（fail-fast）；两处启动路径（外部 redis / 自拉 server）共用。"""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            if client.ping():
                return
        except Exception:  # noqa: BLE001  就绪前 ping 失败属预期
            time.sleep(0.05)
    raise RuntimeError(f"{what} 10s 未就绪")


def decode_member(m):
    """解出 (chunk, seq)：member = 2B 时长桶 + 20B FIFO 序号 + pickle 载荷。

    坏 member（截断/外部误写）返回 (None, None)，由调用方丢弃计数——
    绝不让一条坏消息炸掉 claim/sweep。"""
    try:
        chunk = pickle.loads(m[_MEMBER_HEADER:])
        seq = int(m[2:_MEMBER_HEADER].decode())
    except Exception:  # noqa: BLE001
        return None, None
    if not isinstance(chunk, dict) or "req_id" not in chunk:
        return None, None
    return chunk, seq


class _RedisQueue:
    """两队列的公共骨架：键名构造、spawn 传递（只传键名元数据，连接在子进程
    按需重建）、惰性 redis 连接、入队时间戳、等待 EMA、幂等关闭。
    子类只实现各自的消息语义（FIFO 邮箱 / 优先级认领）。"""

    def __init__(self, maxsize: int = 0, index: int = 0, name: str = ""):
        self._index = int(index)
        self._name = name or f"q{index}"
        self._cap = int(maxsize) if maxsize else 1024
        self._key = f"asr:q:{self._index}:{self._name}"
        self._key_in = self._key + ":in"
        self._key_out = self._key + ":out"
        self._key_avg = self._key + ":avg"   # 近期平均排队 ms（EMA）
        self._client = None

    # ── spawn 传递：只传键名元数据，连接在子进程按需重建 ──
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_client"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._client = None

    def _redis(self):
        if self._client is None:
            self._client = _make_redis_client()
        return self._client

    def _stamp(self, item):
        if isinstance(item, dict):
            item["_q_ts"] = time.time()
        return item

    def _ema_push(self, client, waits_ms) -> None:
        """批量等待样本并入 EMA（近期平均排队 ms）。失败只告警不外抛
        ——心跳/泵线程靠本方法存活，纯指标绝不影响消息交付。"""
        try:
            w = sum(waits_ms) / len(waits_ms)
            cur = float(client.get(self._key_avg) or 0)
            client.set(self._key_avg, w if cur <= 0
                       else (1 - _EMA_ALPHA) * cur + _EMA_ALPHA * w)
        except Exception as exc:  # noqa: BLE001  纯指标
            _qlog.warning("queue %s wait-EMA update failed: %s",
                          self._name, str(exc)[:120])

    def close(self):
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001  幂等
                pass


class InstrumentedQueue(_RedisQueue):
    """跨进程有界队列（Redis RPUSH + LPOP）：结果邮箱等 FIFO 场景。

    put 超时抛 queue.Full、get 超时抛 queue.Empty；depth（qsize）= LLEN。
    崩溃语义：消息常驻 redis-server——worker 死亡队列内消息零丢失；
    坏 pickle 丢弃计数绝不向上抛（泵线程 get() 在 try 之外，抛一次即泵死）。"""

    def _try_push(self, data) -> bool:
        """LLEN 判满 + RPUSH（非原子：超卖 ≤ 并发生产者数、随消费自愈）。"""
        client = self._redis()
        if client.llen(self._key) >= self._cap:
            return False
        client.rpush(self._key, data)   # 尾进 + LPOP 头出 = FIFO
        client.incr(self._key_in)
        return True

    def put(self, item, *args, **kwargs):
        """入队；满时等待至 timeout（None=无限等），超时抛 queue.Full。"""
        timeout = kwargs.get("timeout", args[0] if args else None)
        data = pickle.dumps(self._stamp(item), protocol=pickle.HIGHEST_PROTOCOL)
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            if self._try_push(data):
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise queue.Full()
            time.sleep(0.02)

    def put_nowait(self, item):
        data = pickle.dumps(self._stamp(item), protocol=pickle.HIGHEST_PROTOCOL)
        if not self._try_push(data):
            raise queue.Full()

    def _recv(self, timeout):
        """出队：LPOP + 轮询。连接类错误自动重试（redis 重启窗口，守护拉起）；
        坏 pickle 丢弃该条继续，绝不向上抛。"""
        client = self._redis()
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        first = True
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if not first and remaining is not None and remaining <= 0:
                raise queue.Empty()
            first = False
            try:
                item = client.lpop(self._key)
            except (redis.exceptions.ConnectionError,
                    redis.exceptions.TimeoutError):
                if remaining is not None and remaining <= _REDIS_POLL_S * 2:
                    raise queue.Empty() from None
                time.sleep(0.2)   # redis-server 重启窗口，守护会拉起
                continue
            if item is not None:
                try:
                    client.incr(self._key_out)
                except (redis.exceptions.ConnectionError,
                        redis.exceptions.TimeoutError):
                    pass   # 指标失败不影响消息交付
                try:
                    return pickle.loads(item)
                except Exception as exc:  # noqa: BLE001  坏消息只丢自身
                    _qlog.warning(
                        "queue %s dropped corrupt message (%d bytes): %s",
                        self._name, len(item), str(exc)[:120])
                    continue
            if remaining is not None:
                time.sleep(min(_REDIS_POLL_S, max(0.0, remaining)))
            else:
                time.sleep(_REDIS_POLL_S)

    def _count_get(self, item):
        if isinstance(item, dict):
            ts = item.pop("_q_ts", None)
            if ts is not None:
                w = (time.time() - ts) * 1000
                item["_qw_ms"] = w
                self._ema_push(self._redis(), [w])
        return item

    def get(self, *args, **kwargs):
        timeout = kwargs.get("timeout", args[0] if args else None)
        return self._count_get(self._recv(timeout))

    def get_nowait(self):
        return self._count_get(self._recv(0))

    def qsize(self):
        return self.depth()

    def depth(self):
        return int(self._redis().llen(self._key))

    def stats(self):
        """四读管道一次发（4→1 RTT）：/health 探针与网关准入的热路径。"""
        pipe = self._redis().pipeline(transaction=False)
        pipe.llen(self._key)
        pipe.get(self._key_in)
        pipe.get(self._key_out)
        pipe.get(self._key_avg)
        depth, in_c, out_c, avg = pipe.execute()
        return {"depth": int(depth or 0), "in": int(in_c or 0),
                "out": int(out_c or 0),
                "avg_wait_ms": round(float(avg or 0), 2)}


class AsrClaimQueue(_RedisQueue):
    """ASR 任务的认领式队列（Redis ZSET：score = 优先级类×10^13 + FIFO 序号）。

    语义对齐原统一池 heapq：worker 认领时取 score 最低者——句子（0）永远
    先于 clip（100/10）出队，同优先级先来先做。认领即标记：Lua 原子完成
    弹出 + 写入 per-worker 在途账本 {key}:infl:{worker_id}——worker 死亡
    由 supervisor 按账本精确重派（秒级），慢 worker 永不被 TTL 误重派。

    键：{key}（ZSET 主队列）、{key}:infl:{worker_id}（per-worker 在途）、
    asr:ratt（HASH 重派计数，跨 worker 保留——毒块语义）、asr:seq（INCR
    FIFO 序号）、{key}:in/:out/:avg/:bad。
    """

    def __init__(self, maxsize: int = 0, index: int = 0, name: str = "asr"):
        super().__init__(maxsize, index, name)
        self._key_seq = "asr:seq"
        self._key_ratt = "asr:ratt"
        self._key_cancel = "asr:cancel"

    def cancel_add(self, req_id: str) -> None:
        cancel_add(req_id)

    def is_cancelled(self, req_id: str) -> bool:
        return is_cancelled(req_id)

    def cancelled_set(self, req_ids) -> set:
        """批量断连标记查询（单次 RTT）。返回已取消的 req_id 集。"""
        req_ids = list(req_ids)
        if not req_ids:
            return set()
        pipe = self._redis().pipeline(transaction=False)
        for rid in req_ids:
            pipe.zscore(self._key_cancel, rid)
        scores = pipe.execute()
        return {rid for rid, s in zip(req_ids, scores) if s is not None}

    def _encode(self, task: dict):
        """member = 2B 时长桶 + 20B FIFO 序号 + pickle 载荷；score 编码优先级。
        时长桶（音频样本数 // 32000 ≈ 2s 桶）供认领 Lua 取"与队首差不多长度"
        的凑批（padding 浪费有界）；无音频任务（punc/vad_stream）桶 0。"""
        seq = int(self._redis().incr(self._key_seq))
        payload = pickle.dumps(self._stamp(task), protocol=pickle.HIGHEST_PROTOCOL)
        samples = int((task.get("piece") or {}).get("samples") or 0)
        bucket = min(0xFFFF, samples // _BUCKET_SAMPLES)
        member = struct.pack(">H", bucket) + f"{seq:020d}".encode() + payload
        pri = int(task.get("priority", PRI_CLIP))
        return member, seq, task_score(pri, seq)

    def put(self, task: dict):
        """入队一个任务：容量门（ZCARD 判满轮询等待=背压传导）→ ZADD → INCR :in。"""
        client = self._redis()
        while client.zcard(self._key) >= self._cap:
            time.sleep(_CLAIM_POLL_S)
        member, _seq, score = self._encode(task)
        client.zadd(self._key, {member: score})
        client.incr(self._key_in)
        return member

    def put_nowait(self, task: dict):
        """无阻塞入队：容量满立即 queue.Full。"""
        client = self._redis()
        if client.zcard(self._key) >= self._cap:
            raise queue.Full()
        member, _seq, score = self._encode(task)
        client.zadd(self._key, {member: score})
        client.incr(self._key_in)

    _CLAIM_SCRIPT = """
        local members = redis.call('ZRANGE', KEYS[1], 0, tonumber(ARGV[3]) - 1)
        if #members == 0 then return {} end
        local claimed = {}
        -- 时长桶（member 头 2B）：取与队首同桶的凑批——"差不多长度"，
        -- 批前向 padding 浪费有界（kokoro 类键认领同款）
        local function bk(m)
            return (string.byte(m, 1) or 0) * 256 + (string.byte(m, 2) or 0)
        end
        local ref = bk(members[1])
        local now = ARGV[2]
        for i, m in ipairs(members) do
            if #claimed >= tonumber(ARGV[1]) then break end
            if bk(m) == ref then
                redis.call('ZREM', KEYS[1], m)
                redis.call('ZADD', KEYS[2], now, m)
                claimed[#claimed + 1] = m
            end
        end
        return claimed
    """

    def claim(self, max_batch: int, worker_id: str = ""):
        """认领一批**与队首时长同桶**的任务（≤max_batch 条，Lua 原子弹出 +
        在途标记）。队首 = 最高优先级最老任务（score 最低）——句子先于 clip
        的语义不变；同桶凑批让批前向 padding 有界。返回 [(member, task), ...]。"""
        client = self._redis()
        now = time.time()
        members = client.eval(
            self._CLAIM_SCRIPT, 2, self._key,
            f"{self._key}:infl:{worker_id}",
            max_batch, now, _CLAIM_SCAN_WINDOW)
        out, waits_ms, bad = [], [], []
        for m in members:
            chunk, _seq = decode_member(m)
            if chunk is None:
                bad.append(m)   # 真损坏：丢弃计数，不能让 claim 炸
                continue
            ts = chunk.pop("_q_ts", None)
            if isinstance(ts, (int, float)):
                w = (now - ts) * 1000
                waits_ms.append(w)
                chunk["_qw_ms"] = w
                chunk["_q_enq"] = ts
                chunk["_q_claim"] = now
            out.append((m, chunk))
        # 以下是"认领已经成功"之后的簿记：任何一步失败都不能让本函数抛错
        # ——member 已在在途账本里，调用方只看到异常会重试 claim，这批
        # member 却无人持有（要等 TTL 才重派）。逐段 best-effort。
        if bad:
            try:
                pipe = client.pipeline(transaction=False)
                pipe.zrem(f"{self._key}:infl:{worker_id}", *bad)
                pipe.hdel(self._key_ratt, *bad)
                pipe.incr(self._key_out, len(bad))
                pipe.incr(self._key + ":bad", len(bad))
                pipe.execute()
            except Exception as exc:  # noqa: BLE001
                _qlog.warning("claim bad-member cleanup failed worker=%s n=%d: %s",
                              worker_id, len(bad), str(exc)[:120])
        if waits_ms:
            self._ema_push(client, waits_ms)
        if out:
            try:
                client.incr(self._key_out, len(out))
            except Exception as exc:  # noqa: BLE001  纯指标
                _qlog.warning("claim out counter failed worker=%s: %s",
                              worker_id, str(exc)[:120])
        return out

    _ACK_SCRIPT = """
        local n = 0
        for i = 1, #ARGV do
            if redis.call('ZREM', KEYS[1], ARGV[i]) == 1 then
                redis.call('HDEL', KEYS[2], ARGV[i])
                n = n + 1
            end
        end
        return n
    """

    def ack(self, members, worker_id: str = "") -> int:
        """交付确认：原子「移除在途成功才清重派计数」。幂等（Lua 内 ZREM==1
        才 HDEL），redis 瞬断重试一次，再失败留痕交 TTL sweep 收口。"""
        if not members:
            return 0
        client = self._redis()
        infl_key = f"{self._key}:infl:{worker_id}"
        for attempt in (1, 2):
            try:
                return int(client.eval(self._ACK_SCRIPT, 2, infl_key,
                                       self._key_ratt, *members))
            except Exception as exc:  # noqa: BLE001
                if attempt == 1:
                    _qlog.warning("ack failed worker=%s n=%d (retry once): %s",
                                  worker_id, len(members), str(exc)[:120])
                    time.sleep(0.05)
                else:
                    _qlog.warning("ack failed worker=%s n=%d (left to TTL sweep): %s",
                                  worker_id, len(members), str(exc)[:120])
        return 0

    def _giveup(self, client, m, infl_key, on_giveup):
        """重派超限/坏 member 的放弃收口：计 :bad + 通知 origin 网关（若有
        on_giveup 回调——客户端立即拿到 500 带原因，而不是等 504 超时）。"""
        try:
            client.zrem(infl_key, m)
            client.hdel(self._key_ratt, m)
            client.incr(self._key + ":bad")
        except Exception:  # noqa: BLE001
            pass
        if on_giveup is not None:
            try:
                chunk, _seq = decode_member(m)
                if chunk is not None:
                    on_giveup(chunk)
            except Exception:  # noqa: BLE001  通知失败不阻塞清扫
                pass

    def requeue_worker(self, worker_id: str,
                       max_redispatch: int | None = None,
                       on_giveup=None) -> int:
        """worker 死亡精确重派：在途账本全部 member 回主队列（score 按原
        优先级+原 FIFO 序号复算，保到达序）。max_redispatch=None 不计数；
        >=0 时超限丢弃（毒块防护，on_giveup 收口）。

        **worker 死亡主路径必须计数**（supervisor 调用时传 max_redispatch）
        ——"死亡→重派→再死"若不计数可无限循环（REDISPATCH_MAX 形同虚设）。"""
        client = self._redis()
        infl_key = f"{self._key}:infl:{worker_id}"
        n = 0
        try:
            members = client.zrange(infl_key, 0, -1)
        except Exception:  # noqa: BLE001  redis 瞬断：下轮 sweep 兜底
            return 0
        for m in members:
            if self._redispatch_one(client, infl_key, m, max_redispatch,
                                    on_giveup):
                n += 1
        return n

    def _redispatch_one(self, client, infl_key: str, m,
                        max_redispatch: int | None, on_giveup) -> bool:
        """单 member 重派：毒块/超限 → _giveup（False）；否则回主队列（True）。
        requeue_worker（死亡路径）与 sweep（TTL 兜底路径）共用同一策略——
        毒块计数、重派上限、score 复算只此一份实现。"""
        chunk, seq = decode_member(m)
        if chunk is None:
            self._giveup(client, m, infl_key, on_giveup)
            return False
        if max_redispatch is not None:
            attempts = client.hincrby(self._key_ratt, m, 1)
            if attempts > max_redispatch:
                self._giveup(client, m, infl_key, on_giveup)
                return False
        pri = int(chunk.get("priority", PRI_CLIP))
        client.zadd(self._key, {m: task_score(pri, seq)})
        client.zrem(infl_key, m)
        return True

    def promote(self, req_id, new_priority: int = PRI_PROMOTED) -> int:
        """任务插队：主队列中该请求的全部 clip 改 score（100→10，仍 > 句子）。
        在途（已认领）任务不动——正在算的没必要也不能改优先级。"""
        client = self._redis()
        rid = str(req_id)
        n = 0
        try:
            for m in client.zrange(self._key, 0, -1):
                chunk, seq = decode_member(m)
                if chunk is None or str(chunk.get("req_id")) != rid:
                    continue
                client.zadd(self._key, {m: task_score(new_priority, seq)}, xx=True)
                n += 1
        except Exception:  # noqa: BLE001
            pass
        return n

    def depth(self):
        return int(self._redis().zcard(self._key))

    def stats(self):
        """五读管道一次发（5→1 RTT）。"""
        pipe = self._redis().pipeline(transaction=False)
        pipe.zcard(self._key)
        pipe.get(self._key_in)
        pipe.get(self._key_out)
        pipe.get(self._key + ":bad")
        pipe.get(self._key_avg)
        depth, in_c, out_c, bad, avg = pipe.execute()
        return {"depth": int(depth or 0), "in": int(in_c or 0),
                "out": int(out_c or 0), "bad": int(bad or 0),
                "avg_wait_ms": round(float(avg or 0), 2)}

    def sweep(self, alive_pids: set, now: float, ttl: float,
              max_redispatch: int, on_giveup=None) -> dict:
        """在途重派扫描（supervisor 监控循环调用）。

        重派条件是「worker 已死」（pid 不在 alive_pids）而非「TTL 超时」
        ——慢 worker 永不被误重派；TTL 只兜 supervisor 漏检的极端
        （SWEEP_TTL = 2×STUCK_KILL_S，且看门狗会先杀超时 worker——
        ttl 必须大于活 worker 的最长持握，调小前先核对这两个常数）。"""
        redispatched = gaveup = 0
        client = self._redis()
        client.zremrangebyscore(self._key_cancel, "-inf", now - 600)
        infl_keys = [k for k in client.scan_iter(
            match=f"{self._key}:infl:*", count=256)]
        for infl_key in infl_keys:
            if isinstance(infl_key, bytes):
                infl_key = infl_key.decode()   # redis-py 默认 bytes key
            try:
                pid = int(worker_pid_from_infl_key(infl_key))
            except ValueError:
                continue
            # 死者立即全量重派；活者仅超长 TTL 兜底（正常处理 << ttl）
            if pid in alive_pids:
                members = client.zrangebyscore(infl_key, "-inf", now - ttl,
                                               start=0, num=100)
            else:
                members = client.zrange(infl_key, 0, -1)
            for m in members:
                if self._redispatch_one(client, infl_key, m, max_redispatch,
                                        on_giveup):
                    redispatched += 1
                else:
                    gaveup += 1
        return {"redispatched": redispatched, "gaveup": gaveup}


# ── worker_id 契约（唯一格式化/解析点）──
# 格式 "{role}-{idx}-{pid}"：compute 认领循环生成 → 写进在途账本键；
# supervisor 死亡收口按它重建；sweep 从账本键解析回 pid 判活。
def make_worker_id(role: str, idx: int, pid: int) -> str:
    return f"{role}-{idx}-{pid}"


def worker_pid_from_infl_key(infl_key: str) -> str:
    """"asr:q:0:asr:infl:asr-0-123" → "asr-0-123" 的 pid 段（int() 由调用方判错）。"""
    wid = infl_key.rsplit(":infl:", 1)[-1]
    return wid.rsplit("-", 1)[-1]


# ── 队列等待时间采样（各 worker 每任务回投前推送；/metrics /queue 展示）──


def push_wait_sample(name: str, wait_ms: float) -> None:
    """任务等待时间样本入 redis（queue 名 vad|asr|punc）。失败不影响推理。"""
    try:
        client = shared_redis()
        key = f"asr:wait:{name}"
        client.lpush(key, f"{float(wait_ms):.2f}")
        client.ltrim(key, 0, _WAIT_LIST_MAX)
    except Exception:  # noqa: BLE001  纯观测
        pass


def wait_stats(name: str) -> dict:
    """某队列最近 100 条任务的等待时间统计 {n, avg_ms, p50_ms, p95_ms, max_ms}。
    空窗返回 n=0；读取失败同样 n=0（观测失败不炸端点）。"""
    out = {"n": 0, "avg_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    try:
        raw = shared_redis().lrange(f"asr:wait:{name}", 0, -1)
    except Exception:  # noqa: BLE001
        return out
    vals = []
    for x in raw:
        try:
            vals.append(float(x.decode() if isinstance(x, bytes) else x))
        except ValueError:
            continue
    if not vals:
        return out
    vals.sort()
    out["n"] = len(vals)
    out["avg_ms"] = round(sum(vals) / len(vals), 2)
    out["p50_ms"] = round(vals[len(vals) // 2], 2)
    out["p95_ms"] = round(vals[min(len(vals) - 1, int(len(vals) * 0.95))], 2)
    out["max_ms"] = round(vals[-1], 2)
    return out


def mailbox_depths(queues: dict) -> dict:
    """Mailbox 统计探针：{名字: {depth, ...}}。任何异常一律 None——心跳线程
    靠它活着，这里抛异常会让 worker 心跳停写而被 supervisor 误杀。"""
    out = {}
    for name, q in queues.items():
        if isinstance(q, (list, tuple)):
            for i, sub in enumerate(q):
                out[f"{name}[{i}]"] = _queue_stats(sub)
        else:
            out[name] = _queue_stats(q)
    return out


def _queue_stats(q) -> dict:
    try:
        if isinstance(q, (InstrumentedQueue, AsrClaimQueue)):
            return q.stats()
        return {"depth": q.qsize()}
    except Exception:  # noqa: BLE001  平台不支持/队列已关闭 → unknown
        return {"depth": None}
