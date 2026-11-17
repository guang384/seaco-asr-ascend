"""跨进程消息协议：类型常量 + 构造函数（单一事实来源）。

流水线全部跨进程队列消息的字段清单集中在此：任何一端要读的字段，
必须由这里的构造函数保证存在（防字段漂移断管）。

消息一览（生产者 → 队列 → 消费者）：
  gateway → vad/asr/punc_queue(ZSET)  *_TASK   推理任务（score 编码优先级：
                                               句子/流式步进 0 < 插队 10 < 批量 100）
  gateway → asr:cancel(ZSET)         ──(非消息)── 超时/断连取消标记（score=时刻）：
                                               worker 认领后、推理前检查跳过
  worker  → {queue}:infl:{id}(ZSET)  ──(非消息)── 认领即标记（at-least-once）：
                                               worker 死亡 → requeue_worker 精确重派
  worker  → gateway_queues[o]        *_RESULT  单任务结果/错误（按 origin 直投受理
                                               网关；网关按 (req_id, stage, idx)
                                               一步一 future 收齐聚合）

音频线格式：SHM 块内 PCM16 int16（<i2），读侧 /32767 还原 float32
（16bit 源无损；高位深源量化噪声 -96dB，远低于语音噪声底）。
SHM 块生命周期：gateway 创建写入 → worker 按名 attach 读取（close 不
unlink）→ gateway 在请求终态（收齐/超时/失败）统一 unlink。块内偏移
由 piece 描述符携带（{shm_name, offset, samples}），消息本身零音频拷贝。
"""

# ── 消息类型常量（gateway 结果队列里多态分派用）──
ASR_TASK = "asr_task"
ASR_RESULT = "asr_result"
VAD_TASK = "vad_task"
VAD_RESULT = "vad_result"
PUNC_TASK = "punc_task"
PUNC_RESULT = "punc_result"

# ── 任务种类（vad/punc/asr 三队列通用优先级语义）──
KIND_SENT = "sent"    # WS 流式句尾句子/流式步进（优先级 0，延迟敏感）
KIND_CLIP = "clip"    # REST 拆段 clip/整段 VAD/全文标点（优先级 100/10，吞吐型）
KIND_VAD_BATCH = "vad_batch"    # 整段音频 VAD 切段（REST）
KIND_VAD_STREAM = "vad_stream"  # 流式 VAD 步进（WS，cache 随消息往返，worker 无状态）

# ── 结果投递超时：目标入口卡死时丢弃本条，不冻结全部工人
#（该请求由入口侧 WAIT_TIMEOUT 超时兜底）──
RESULT_PUT_TIMEOUT_S = 30.0


def asr_task(req_id: str, idx: int, origin: int, kind: str, piece: dict,
             hotword: str | None = None, priority: int = 100) -> dict:
    """gateway → asr_queue（ZSET 认领式）。

    piece = {shm_name, offset, samples}：音频在 SHM 块内的一段（PCM16）。
    origin 是 ASR_RESULT 的回投网关序号。
    priority：句子 0 / 插队 clip 10 / 普通 clip 100——入队 score = 优先级
    × PRIORITY_BASE + FIFO 序号，认领取最低分。"""
    return {"type": ASR_TASK, "req_id": str(req_id), "idx": int(idx),
            "origin": int(origin), "kind": kind, "piece": piece,
            "hotword": hotword, "priority": int(priority)}


def asr_result(req_id: str, idx: int, kind: str = KIND_CLIP,
               text: str | None = None, error: str | None = None,
               timings: dict | None = None) -> dict:
    """worker → gateway_queues[origin]。单任务终态：text 或 error 二选一。
    timings：可选耗时（毫秒）{"q_wait", "infer", "done_at"}（done_at 为
    回投绝对时刻秒）；None 不落消息（缺字段消费端按 0 兜底）。"""
    msg = {"type": ASR_RESULT, "req_id": str(req_id), "idx": int(idx),
           "kind": kind, "text": text, "error": error}
    if timings is not None:
        msg["timings"] = timings
    return msg


def vad_task(req_id: str, idx: int, origin: int, kind: str,
             piece: dict | None = None, chunk=None, cache=None,
             is_final: bool = False, chunk_size: int = 200,
             max_end_silence_time: int = 800, priority: int = 100) -> dict:
    """gateway → vad_queue（ZSET 认领式）。

    kind=vad_batch：piece={shm_name, offset, samples}（整段音频，REST 拆句）；
    kind=vad_stream：chunk=np.ndarray(200ms float32) + cache=流式状态机
    （随消息往返，worker 无状态；is_final 触发尾部闭合）。"""
    msg = {"type": VAD_TASK, "req_id": str(req_id), "idx": int(idx),
           "origin": int(origin), "kind": kind,
           "priority": int(priority), "is_final": bool(is_final),
           "chunk_size": int(chunk_size),
           "max_end_silence_time": int(max_end_silence_time)}
    if piece is not None:
        msg["piece"] = piece
    if chunk is not None:
        msg["chunk"] = chunk
    if cache is not None:
        msg["cache"] = cache
    return msg


def vad_result(req_id: str, idx: int, segments: list,
               cache=None, error: str | None = None,
               done: bool = True, timings: dict | None = None) -> dict:
    """vad worker → gateway_queues[origin]。segments=[[s_ms,e_ms],...]（增量段）；
    cache 仅 vad_stream 往返（worker mutate 后回传，gateway 存回 Session）。

    done=False 为**边切边发**的中间结果：vad_batch 任务边走边吐新完成段，
    gateway 收到即派发 ASR clip（识别与切段重叠）；done=True 为终态（可无段）。"""
    msg = {"type": VAD_RESULT, "req_id": str(req_id), "idx": int(idx),
           "segments": segments or [],
           "cache": cache, "error": error, "done": bool(done)}
    if timings is not None:
        msg["timings"] = timings
    return msg


def punc_task(req_id: str, idx: int, origin: int, text: str,
              priority: int = 100) -> dict:
    """gateway → punc_queue（ZSET 认领式）：单段文本补标点。"""
    return {"type": PUNC_TASK, "req_id": str(req_id), "idx": int(idx),
            "origin": int(origin), "text": text, "priority": int(priority)}


def punc_result(req_id: str, idx: int, text: str | None,
                error: str | None = None, timings: dict | None = None) -> dict:
    """punc worker → gateway_queues[origin]。"""
    msg = {"type": PUNC_RESULT, "req_id": str(req_id), "idx": int(idx),
           "text": text, "error": error}
    if timings is not None:
        msg["timings"] = timings
    return msg


class VadSegCollector:
    """FSMN-VAD 流式事件 → 完成段状态机（跨进程唯一实现）。

    事件协议（funasr 流式 VAD 输出，[beg_ms, end_ms]，-1 为空位）：
    [beg,-1] 段开始 / [-1,end] 段结束 / [beg,end] 完整段（final 强制闭合、
    超长强切）。REST 整段切段（compute vad worker）与 WS 流式会话
    （gateway Session）共用本类——切段语义一份实现，两形态零漂移。
    坐标为全局毫秒（随 cache 递增）；cur_start 跨步持续（会话内未闭合段）。"""

    def __init__(self):
        self.cur_start = None

    def feed(self, events) -> list:
        """吃一步事件，返回本步新完成的段 [[s_ms, e_ms], ...]。"""
        out = []
        for seg in events or []:
            if len(seg) != 2:
                continue
            s, e = int(seg[0]), int(seg[1])
            if s != -1 and e != -1:       # 完整段（final 强制闭合 / 超长强切）
                out.append([s, e])
            elif s != -1:                 # [beg, -1]：段开始
                self.cur_start = s
            elif e != -1:                 # [-1, end]：段结束
                if self.cur_start is not None:
                    out.append([self.cur_start, e])
                    self.cur_start = None
        return out


# 结果投递丢弃计数（at-most-once 设计内的受控丢弃）：各 worker 进程独立
# 计数，随心跳 stats 上报。键预置、主循环线程递增——无跨线程竞态。
_DROPS = {"result": 0}


def drop_count() -> int:
    """本进程累计丢弃的结果消息数（put_result 失败等）。"""
    return _DROPS["result"]


def put_result(q, msg: dict, log=None) -> bool:
    """带超时的结果投递：目标入口阻塞时宁可丢这一条——消息级 at-most-once，
    该请求由入口侧超时兜底，绝不让单个入口的阻塞冻结全部工人。
    （worker 死亡导致的在途丢失是另一条路径——supervisor 的
    requeue_worker/sweep 精确重派兜底，见 AsrClaimQueue。）"""
    try:
        q.put(msg, timeout=RESULT_PUT_TIMEOUT_S)
        return True
    except Exception as exc:  # noqa: BLE001  Full（入口积压）或队列关闭
        _DROPS["result"] += 1
        if log is not None:
            log.warning("result delivery failed (dropped, req=%s type=%s): %s",
                        msg.get("req_id"), msg.get("type"), str(exc)[:120])
        return False
