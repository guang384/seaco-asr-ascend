"""gateway 进程：HTTP/WS 接入 + 音频解码 + 流水编排（SO_REUSEPORT 多进程）。

拓扑（由 supervisor.py 编排启动，不可直接运行）——gateway **无模型**，
推理全在三类 worker 进程（各自独立队列，等待时间进指标）：

    REST:  解码 → SHM 块 → [vad_queue 整段切段] → [asr_queue 逐 clip]
           → [punc_queue 全文标点] → 响应
    WS:    帧缓冲 → [vad_queue 流式步进(cache 往返)] → 句尾
           → [asr_queue 句子] → [punc_queue 句尾标点] → 逐句 JSON

统一优先级语义（三队列同构）：流式/句尾（0）永远先于批量（100，插队 10）
被认领。SHM 块生命周期：gateway 创建写入 → worker 按名 attach 读取
（close 不 unlink）→ 请求终态统一 unlink；孤儿块超龄清道夫兜底。

双模式输入（同一端点按 Content-Type 分流）：
    multipart/form-data → file 字段（OpenAI 兼容）
    application/json    → {"audio_base64": "..."(支持 data URI)，"hotword": "..."}
"""
import asyncio
import base64
import binascii
import collections
import concurrent.futures
import json
import os
import socket
import threading
import time
import uuid

import numpy as np

import audio
import ipc
import proc_common
import protocol
from audio import SAMPLE_RATE
from audio import decode_audio as _decode_audio   # 模块别名：测试经 gateway._decode_audio 打桩
from ipc import PRI_CLIP, PRI_PROMOTED, PRI_SENT
from proc_common import env_float, env_int, setup_logging

log = setup_logging("gateway")

# ── 配置 ──
MODE = (os.getenv("MODE") or "mixed").strip().lower()
HEALTH_DIR = os.getenv("ASR_HEALTH_DIR", "/tmp/asr-health")
IMAGE_VERSION = os.getenv("IMAGE_VERSION", "dev")
CHUNK_MS = 200                     # 流式 VAD 步进（ms）
MAX_END_SILENCE_MS = 800           # 尾静音判句阈值（ms）
MAX_AUDIO_MB = env_int("MAX_AUDIO_MB", 200)
WAIT_TIMEOUT_S = env_float("WAIT_TIMEOUT_S", 120.0, minimum=1.0)
MAX_PENDING = env_int("MAX_PENDING_REQUESTS", 8, minimum=1)
MAX_INFLIGHT = env_int("MAX_INFLIGHT_TASKS", 512, minimum=1)
SHM_ORPHAN_MAX_AGE_S = env_float("SHM_ORPHAN_MAX_AGE_S", 300.0, minimum=30.0)
HB_STALE_S = env_float("HEARTBEAT_STALE_S", 30.0, minimum=15.0)
QUEUE_NAMES = ("vad", "asr", "punc")
METRICS_ENABLED = env_int("METRICS_ENABLED", 1) != 0   # /metrics 开关（0=404）

# 容量估算常量（先验；滚动样本 ≥10 条后被实测覆盖）
VAD_ROUTES_PER_WORKER = 60   # 每 VAD worker 理论流式路数
SENT_RATE_PER_ROUTE = 0.2    # 每路出句速率（句/s）
SENT_PRIOR_S = 0.5           # 先验：单句占用秒
BATCH_PRIOR_X = 15.0         # 先验：单实例批量音频倍速

# ── 运行时状态（gateway_main 注入队列句柄后可用）──
_q = {}                      # {"vad"/"asr"/"punc": AsrClaimQueue}
_result_q = None             # 本入口结果邮箱（worker 按 origin 直投）
_worker_origin = 0
_main_loop = None            # 服务事件循环（结果泵回调）
_pending = {}                # (req_id, stage, idx) → {"fut", "on_segments"?}
_pending_lock = threading.Lock()
_req_states = {}             # req_id → {"shm": block}（SHM 块生命周期）
_active = {}                 # req_id → brief（/queue 观测；终态移除）
_active_lock = threading.Lock()

REST_GATE = threading.BoundedSemaphore(MAX_PENDING)   # 受理级闸门（每 gateway）

# 统计（进程内累计，随心跳上报、跨 gateway 聚合）
_entry_stats = {"requests": 0, "successes": 0, "timeouts": 0, "failed": 0,
                "rejections": 0, "server_5xx": 0, "sentences": 0,
                "audio_total_s": 0.0, "wall_total_s": 0.0}
_recent = collections.deque(maxlen=10)        # 最近任务 brief（含终态）
_recent_failures = collections.deque(maxlen=3)
_ws_pool = None           # WS feed 线程池（几十路并发，默认池不够）
_prep_pool = None         # REST 解码线程池

_STAGE_OF = {protocol.VAD_RESULT: "vad", protocol.ASR_RESULT: "asr",
             protocol.PUNC_RESULT: "punc"}


# ---------------- 阶段调用（req_id+stage future 注册/完成/等待） ----------------

def _new_req_id() -> str:
    """全局任务 id（redis INCR，跨 gateway 唯一）；REST 响应 job_id 同源。"""
    try:
        return str(int(ipc.shared_redis().incr("asr:jid")))
    except Exception:  # noqa: BLE001  redis 瞬断：本地 uuid 兜底（仍唯一）
        return uuid.uuid4().hex[:12]


def _register_stage(req_id: str, stage: str, fut, idx: int = 0,
                    on_segments=None) -> str:
    """登记阶段 future：key=(req_id, stage, idx)。vad 阶段可带 on_segments
    回调（边切边发：增量段在回调里即刻派发 clip）。"""
    key = (str(req_id), stage, int(idx))
    with _pending_lock:
        _pending[key] = {"fut": fut, "on_segments": on_segments}
    return key


def _finish_stage(key) -> dict | None:
    with _pending_lock:
        return _pending.pop(key, None)


def _clear_pending(req_id: str) -> None:
    """清掉该请求的全部阶段 future（终态收口；键形如 (req_id, stage, idx)）。"""
    rid = str(req_id)
    with _pending_lock:
        for key in [k for k in _pending if k[0] == rid]:
            _pending.pop(key, None)


def _complete_stage(st: dict, res: dict) -> None:
    fut = st["fut"]
    try:
        if not fut.done():
            fut.set_result(res)
    except Exception:  # noqa: BLE001  future 已取消/超时：忽略
        pass


def _handle_result(msg: dict) -> None:
    """结果消息分发（事件循环线程内执行，经结果泵递交）。

    边切边发：vad_batch 的 done=False 增量消息触发 on_segments 回调
    （gateway 立即派发该段 ASR clip），不完成 future；done=True 才收尾。
    其余结果按 (req_id, stage, idx) 一步一 future；孤儿结果静默丢。"""
    stage = _STAGE_OF.get(msg.get("type"))
    if stage is None:
        return
    key = (str(msg.get("req_id")), stage, int(msg.get("idx", 0)))
    st = _pending.get(key)
    if st is None:
        return
    if stage == "vad":
        cb = st.get("on_segments")
        if cb is not None:
            try:
                cb(msg.get("segments") or [])   # 增量段与终态段都过回调（去重在回调内）
            except Exception:  # noqa: BLE001  回调异常不杀泵
                log.exception("vad on_segments callback failed")
        if not msg.get("done", True):
            return                              # 增量消息：future 不动，等终态
    _complete_stage(st, {"text": msg.get("text"), "error": msg.get("error"),
                         "raw": msg})


def _punc_sender(req_id: str, text: str, priority: int):
    """punc 阶段入队闭包（WS 句尾 PRI_SENT / REST 全文 PRI_CLIP 共用）。"""

    def _send():
        _q["punc"].put_nowait(protocol.punc_task(
            req_id, 0, _worker_origin, text, priority=priority))
    return _send


def _pump_once(result_q) -> bool:
    """单次取消息 + 递交；任何异常都不外抛（泵线程绝不能死——泵一死所有
    后续请求 504，而 /health 可能还是绿的）。返回 False = 本轮取失败。"""
    try:
        item = result_q.get()
    except Exception:  # noqa: BLE001
        log.exception("result pump dequeue failed (retry)")
        return False
    try:
        _main_loop.call_soon_threadsafe(_handle_result, item)
    except Exception:  # noqa: BLE001
        log.exception("result pump delivery failed (message dropped)")
    return True


def _result_pump(result_q) -> None:
    """独立线程：把结果邮箱的事件递交给服务事件循环。"""
    while True:
        if _main_loop is None or _main_loop.is_closed():
            time.sleep(0.05)
            continue
        if not _pump_once(result_q):
            time.sleep(0.05)


def _stage_sync(req_id: str, stage: str, enqueue, idx: int = 0,
                timeout: float = 60.0):
    """同步阶段调用（WS feed 线程）：注册 future → 入队 → 等结果。
    卡超 timeout 先打现场日志再继续等（观测兜底，不中断会话）。"""
    fut = concurrent.futures.Future()
    key = _register_stage(req_id, stage, fut, idx=idx)
    try:
        enqueue()
    except Exception:
        _finish_stage(key)
        raise
    try:
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            log.warning("阶段 %s [%s] 已等 %.0fs：队列 %s（心跳久且不动 = "
                        "worker 卡在推理）", stage, req_id, timeout,
                        {n: _q[n].depth() for n in QUEUE_NAMES})
            return fut.result()
    finally:
        _finish_stage(key)


async def _stage_async(req_id: str, stage: str, enqueue, deadline: float,
                       idx: int = 0):
    """异步阶段调用（REST）：注册 future → 入队 → 等结果（截止线=请求总预算）。"""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    key = _register_stage(req_id, stage, fut, idx=idx)
    try:
        enqueue()
    except Exception:
        _finish_stage(key)
        raise
    try:
        remaining = max(0.1, deadline - time.perf_counter())
        return await asyncio.wait_for(fut, timeout=remaining)
    finally:
        _finish_stage(key)


# ---------------- WS 流式会话（VAD cache 随消息往返，worker 无状态） ----------------

class Session:
    """单连接会话：音频缓冲 + VAD 状态机（VAD 推理在 vad-worker 进程）。

    二进制帧兼容两种格式（ingest_bytes 自动识别）：
    - wav 文件流：首帧含 RIFF 头即解析（任意采样率/声道自动转换），后续帧为数据
    - 裸 PCM：16kHz/16bit/mono（默认）
    """

    def __init__(self):
        self.cache = {}         # FSMN-VAD 流式状态（vad_stream 消息往返）
        self.audio = []         # float32 全局缓冲，按段切给 ASR
        self._col = protocol.VadSegCollector()   # VAD 事件状态机（与 vad worker 同一实现）
        self.base_samp = 0      # 缓冲头部已丢弃的样本数
        self._safe_ms = 0       # 最近一个已完成段的结束位置（全局 ms）
        self.wav = None         # 已解析 wav 头 fmt；None = 裸 PCM 模式
        self.pending = b""      # wav 头跨帧不完整时暂存
        self._buf = np.zeros(0, dtype=np.float32)   # 16k 对齐缓冲
        self.inflight = None    # 在途句子/步进 req_id（断连取消用）
        self.active_key = None  # _active 观测条目键（ws_stream 登记，断连移除）

    def _slice(self, s_ms: int, e_ms: int):
        """从全局缓冲切出句子波形；<100ms 碎片返回 None 直接丢弃"""
        i0 = s_ms * 16 - self.base_samp
        i1 = e_ms * 16 - self.base_samp
        wave = np.concatenate(self.audio)[max(i0, 0):i1]
        return wave if len(wave) >= SAMPLE_RATE // 10 else None

    def ingest_bytes(self, raw: bytes):
        """WS 二进制帧 → int16@16k/mono ndarray；wav 头不完整时返回 None 等下一帧"""
        if self.wav is None:
            buf = self.pending + raw
            self.pending = b""
            if buf[:4] != b"RIFF":
                return np.frombuffer(buf, dtype=np.int16)      # 裸 PCM（16k 假定）
            fmt, off = audio.parse_wav(buf)
            if fmt is None:
                self.pending = buf                             # 头未收全
                return None
            self.wav = fmt
            data = buf[off:]
        else:
            data = raw
        return audio.to16k(audio.decode_wav(data, self.wav), self.wav["fs"])

    def _vad_step(self, wave: np.ndarray, is_final: bool) -> list:
        """远程 VAD 步进（vad-worker 进程）：chunk+cache 随消息往返；
        返回本次完成的句子 [(s_ms, e_ms, wave), ...]。

        段完成判定走 protocol.VadSegCollector（与 vad worker 同一实现）；
        波形切片（_slice/_trim/_safe_ms）在本进程。"""
        req_id = _new_req_id()
        self.inflight = req_id

        def _send():
            _q["vad"].put_nowait(protocol.vad_task(
                req_id, 0, _worker_origin, protocol.KIND_VAD_STREAM,
                chunk=wave, cache=self.cache, is_final=is_final,
                chunk_size=CHUNK_MS, max_end_silence_time=MAX_END_SILENCE_MS,
                priority=PRI_SENT))

        res = _stage_sync(req_id, "vad", _send)
        self.inflight = None
        if res.get("error"):
            log.warning("vad_stream 失败 [%s]: %s", req_id, res["error"])
            return []
        if res["raw"].get("cache") is not None:
            self.cache = res["raw"]["cache"]
        out = []
        for s_ms, e_ms in self._col.feed(res["raw"].get("segments")):
            w = self._slice(s_ms, e_ms)
            if w is not None:
                out.append((s_ms, e_ms, w))
            self._safe_ms = e_ms
        cur = self._col.cur_start
        self._trim(cur if cur is not None else self._safe_ms)
        return out

    def _trim(self, keep_ms: int):
        """丢弃 keep_ms（全局坐标）之前的缓冲：已完成段不会再被引用。
        长会话内存从 ~230MB/小时 降到秒级残留。"""
        target = keep_ms * (SAMPLE_RATE // 1000)
        n = target - self.base_samp
        if n < SAMPLE_RATE // 10:
            return
        while self.audio and len(self.audio[0]) <= n:
            n -= len(self.audio.pop(0))
        if n > 0 and self.audio:
            self.audio[0] = self.audio[0][n:]
            n = 0
        self.base_samp = target - n

    def feed(self, pcm: np.ndarray, is_final: bool) -> list:
        """喂入 int16 PCM，返回 [(begin_ms, end_ms, text), ...]（本次新完成句子）。

        两段式（几十路并发的关键）：
        1. 缓冲 + 远程 VAD 步进（vad_queue，句子/步进优先级 0）
        2. 逐句 asr_queue → punc_queue（优先级 0），等待期间不持任何状态锁"""
        f32 = pcm.astype(np.float32) / 32768.0
        self.audio.append(f32)
        self._buf = np.concatenate([self._buf, f32])
        if self.active_key is not None:     # 观测：流式会话累计音频秒（/queue 可见）
            with _active_lock:
                st = _active.get(self.active_key)
                if st is not None:
                    st["audio_s"] = round(
                        st["audio_s"] + len(pcm) / SAMPLE_RATE, 2)
        block = SAMPLE_RATE * CHUNK_MS // 1000
        segs = []
        while self._buf.size >= block:
            segs += self._vad_step(self._buf[:block], False)
            self._buf = self._buf[block:]
        if is_final:
            if self._buf.size:
                segs += self._vad_step(self._buf, True)
                self._buf = self._buf[:0]
            else:       # 补 1 个零样本触发 VAD flush（源码约定：final 帧把 prev_samples 走完）
                segs += self._vad_step(np.zeros(1, np.float32), True)
        out = []
        for s_ms, e_ms, wave in segs:
            text = self._recognize(wave)
            out.append((s_ms, e_ms, text))
        return out

    def _recognize(self, wave: np.ndarray) -> str:
        """单句两阶段：asr_queue 句子推理（优先级 0）→ punc_queue 句尾标点。"""
        req_id = _new_req_id()
        self.inflight = req_id
        try:
            block = _register_req_audio(req_id, wave)
            piece = dict(block["segs"][0], shm_name=block["shm_name"])

            def _send_asr():
                _q["asr"].put_nowait(protocol.asr_task(
                    req_id, 0, _worker_origin, protocol.KIND_SENT, piece,
                    priority=PRI_SENT))
            res = _stage_sync(req_id, "asr", _send_asr)
            if res.get("error"):
                return ""
            text = res.get("text") or ""
            if text:
                pres = _stage_sync(req_id, "punc",
                                   _punc_sender(req_id, text, PRI_SENT))
                if not pres.get("error"):
                    text = pres.get("text") or text
            _entry_stats["sentences"] += 1
            return text
        finally:
            self.inflight = None
            _finish_req(req_id)


# ---------------- SHM 块生命周期（gateway 唯一清理者） ----------------

def _write_shm_block(pieces: list) -> dict:
    """float32 波形列表 → 单 SHM 块（PCM16 int16 线格式）+ 段偏移表。
    返回 {"shm_name", "segs": [{idx, offset, samples}], "_handle"}。
    写侧句柄随请求状态持有到终态（Windows 命名映射随最后句柄关闭而销毁，
    过早 close 会让 worker attach 不到）；unlink 归 _finish_req。"""
    from multiprocessing import shared_memory
    total = sum(len(p) for p in pieces)
    shm = shared_memory.SharedMemory(
        create=True, size=max(total * 2, 1), name=f"asr_{uuid.uuid4().hex[:12]}")
    segs, off = [], 0
    for i, p in enumerate(pieces):
        pcm = np.clip(p * 32767.0, -32768, 32767).astype("<i2")
        shm.buf[off:off + pcm.nbytes] = pcm.tobytes()
        segs.append({"idx": i, "offset": off, "samples": len(p)})
        off += pcm.nbytes
    return {"shm_name": shm.name, "segs": segs, "_handle": shm}


def _unlink_block(block) -> None:
    """SHM 块清理：close + unlink（幂等）。写侧句柄在这里关闭。"""
    if not block:
        return
    h = block.pop("_handle", None)
    if h is not None:
        try:
            h.close()
        except Exception:  # noqa: BLE001
            pass
    try:
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(name=block["shm_name"])
        shm.close()
        shm.unlink()
    except Exception:  # noqa: BLE001  已被清理/名字不存在：幂等
        pass


def _shm_stats() -> dict | None:
    """/dev/shm 使用情况（MB + 水位）；不可用（非 POSIX/Windows）返回 None。"""
    u = proc_common.shm_usage()
    if u is None:
        return None
    return {"total_mb": u["total_mb"], "used_mb": u["used_mb"], "pct": u["pct"]}


def _shm_used_pct() -> float:
    """/dev/shm 水位（0-100）；读不了返回 -1。"""
    u = proc_common.shm_usage()
    return -1.0 if u is None else u["pct"]


def _sweep_orphan_shm() -> int:
    """回收无主 SHM 块（asr_* 前缀且超龄）：worker 被 SIGBUS 杀死时来不及
    unlink 的半成品块永久占 /dev/shm 的兜底。非 POSIX 返回 0。"""
    try:
        names = os.listdir("/dev/shm")
    except OSError:
        return 0
    cutoff = time.time() - SHM_ORPHAN_MAX_AGE_S
    n = 0
    for nm in names:
        if not nm.startswith("asr_"):
            continue
        path = os.path.join("/dev/shm", nm)
        try:
            if os.path.getmtime(path) > cutoff:
                continue
            os.unlink(path)
            n += 1
        except OSError:
            continue
    if n:
        log.warning("swept %d orphan SHM blocks (age > %.0fs)",
                    n, SHM_ORPHAN_MAX_AGE_S)
    return n


def _register_req_audio(req_id: str, wave: np.ndarray) -> dict:
    """写 SHM 块 + 登记请求状态（REST/WS 共用；unlink 归 _finish_req 终态收口）。"""
    block = _write_shm_block([wave])
    _req_states[req_id] = {"shm": block}
    return block


def _finish_req(req_id: str) -> dict | None:
    """请求终态收口（幂等）：释放 SHM、出 _active、回 brief。"""
    st = _req_states.pop(req_id, None)
    if st is not None:
        _unlink_block(st.get("shm"))
    with _active_lock:
        return _active.pop(req_id, None)


# ---------------- FastAPI 应用（业务端口：REST + WS + /health + /metrics） ----------------

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402

app = FastAPI(title="ASR Server", docs_url="/docs")


def _reject(code: int, reason: str, detail: str, retry_after: str = "2"):
    raise HTTPException(status_code=code, detail=detail,
                        headers={"Retry-After": retry_after,
                                 "X-Reject-Reason": reason})


async def _admission() -> None:
    """受理闸门（顺序检查，满即 429 + X-Reject-Reason）：
    shm_pressure（/dev/shm ≥90%）> queue_full（asr 积压）> gate（在途请求）。"""
    pct = _shm_used_pct()
    if pct >= 90:
        _reject(429, "shm_pressure", f"/dev/shm {pct:.0f}% full", "1")
    try:
        depth = _q["asr"].depth()
    except Exception:  # noqa: BLE001
        depth = 0
    if depth >= MAX_INFLIGHT:
        _reject(429, "queue_full", f"queue full: {depth} tasks in flight")
    if not REST_GATE.acquire(blocking=False):
        _reject(429, "queue_full",
                f"too many requests in flight: {MAX_PENDING}")


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    """完整音频转写（双模式输入，响应 {"text", "job_id"}）。

    multipart/form-data：file 字段（OpenAI 兼容；model/language 忽略）
    application/json：{"audio_base64"(支持 data URI 前缀)，"hotword"，"filename"?}
    """
    if MODE == "stream":
        raise HTTPException(status_code=503, detail="no batch service (MODE=stream)")
    ct = (request.headers.get("content-type") or "").lower()
    if ct.startswith("multipart/"):
        try:
            form = await request.form()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"bad form: {exc}")
        f = form.get("file")
        if f is None or not hasattr(f, "read"):
            raise HTTPException(status_code=400, detail="missing 'file' field")
        data = await f.read()
        name = getattr(f, "filename", None) or "upload"
        hotword = form.get("hotword") or None
    elif ct.startswith("application/json"):
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"bad json: {exc}")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="json body must be an object")
        b64 = body.get("audio_base64") or ""
        if not isinstance(b64, str) or not b64.strip():
            raise HTTPException(status_code=400,
                                detail="missing 'audio_base64' field")
        b64 = b64.strip()
        if b64.startswith("data:"):          # data URI：剥前缀
            b64 = b64.split(",", 1)[-1]
        try:
            data = base64.b64decode("".join(b64.split()), validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="invalid base64 in audio_base64")
        name = body.get("filename") or "base64"
        hotword = body.get("hotword") or None
    else:
        raise HTTPException(status_code=415,
                            detail="unsupported content-type: use multipart/form-data"
                                   " (file) or application/json (audio_base64)")
    if not data:
        return {"text": "", "error": "empty file"}
    if len(data) > MAX_AUDIO_MB * 2**20:
        raise HTTPException(status_code=413,
                            detail=f"audio too large: >{MAX_AUDIO_MB}MB")
    await _admission()          # 429 路径不占 gate（acquire 失败即抛）
    t0 = time.perf_counter()
    try:
        return await _run_transcription(data, name, hotword, t0)
    finally:
        REST_GATE.release()


async def _run_transcription(data: bytes, name: str, hotword, t0: float):
    """REST 三级流水（**边切边发**）：解码 → vad_queue 整段切段（增量段随到随
    派发 clip）→ asr_queue 并行识别 → punc_queue 全文标点。
    切段与识别重叠：首段不等整段 VAD 完成；单截止线（WAIT_TIMEOUT_S）覆盖全流水。"""
    loop = asyncio.get_running_loop()
    deadline = t0 + WAIT_TIMEOUT_S

    def _decode():
        return _decode_audio(data)

    try:
        wave = await loop.run_in_executor(_prep_pool, _decode)
    except Exception as exc:  # noqa: BLE001  解码失败（坏音频）
        _entry_stats["failed"] += 1
        raise HTTPException(status_code=400, detail=f"decode failed: {str(exc)[:200]}")
    audio_s = len(wave) / SAMPLE_RATE
    req_id = _new_req_id()
    job_id = int(req_id) if req_id.isdigit() else 0
    block = _register_req_audio(req_id, wave)
    with _active_lock:
        _active[req_id] = {"id": job_id, "name": name[:80],
                           "audio_s": round(audio_s, 2), "kind": "clip",
                           "started_at": time.time()}
    _entry_stats["requests"] += 1
    # 边切边发状态：增量段回调在事件循环线程内派发 clip（每 clip 一个 future）
    segs_seen = set()          # (s,e) 去重：worker 重派重发的段不重复算
    clip_futs = []             # (idx, future) 提交顺序 = 音频顺序

    def on_segments(segments):
        for s, e in segments:
            s, e = int(s), int(e)
            if e <= s or (s, e) in segs_seen:
                continue
            segs_seen.add((s, e))
            idx = len(clip_futs)
            piece = {"shm_name": block["shm_name"], "offset": s * 16 * 2,
                     "samples": (e - s) * 16}
            cfut = loop.create_future()
            _register_stage(req_id, "asr", cfut, idx=idx)
            clip_futs.append((idx, cfut))
            try:
                _q["asr"].put_nowait(protocol.asr_task(
                    req_id, idx, _worker_origin, protocol.KIND_CLIP, piece,
                    hotword=hotword, priority=PRI_CLIP))
            except Exception as exc:  # noqa: BLE001  入队失败：该 clip 立即报错
                _complete_stage({"fut": cfut},
                                {"text": None, "error": str(exc)[:120],
                                 "raw": {}})
    try:
        # ① 整段 VAD（边切边发：on_segments 随增量段派发 clip）
        piece0 = dict(block["segs"][0], shm_name=block["shm_name"])
        vad_fut = loop.create_future()
        _register_stage(req_id, "vad", vad_fut, idx=0,
                        on_segments=on_segments)

        def _send_vad():
            _q["vad"].put_nowait(protocol.vad_task(
                req_id, 0, _worker_origin, protocol.KIND_VAD_BATCH,
                piece=piece0, priority=PRI_CLIP))
        try:
            _send_vad()
            vres = await asyncio.wait_for(
                vad_fut, timeout=max(0.1, deadline - time.perf_counter()))
        except (asyncio.TimeoutError, TimeoutError):
            raise _timeout_error(req_id, name)
        finally:
            _finish_stage((req_id, "vad", 0))
        if vres.get("error"):
            raise HTTPException(status_code=500,
                                detail=f"vad failed: {vres['error']}")
        # ② 等全部 clip（增量派发，收齐按 idx 顺序拼文）
        if not clip_futs:
            _entry_stats["successes"] += 1
            _entry_stats["audio_total_s"] += audio_s
            return {"text": "", "job_id": 0}   # 无语音段：文档口径 job_id=0
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[f for _i, f in clip_futs]),
                timeout=max(0.1, deadline - time.perf_counter()))
        except (asyncio.TimeoutError, TimeoutError):
            raise _timeout_error(req_id, name)
        for r in results:
            if r.get("error"):
                raise HTTPException(status_code=500, detail=r["error"])
        text = "".join(r.get("text") or "" for r in results)
        # ③ 全文标点（失败保留原文）
        if text:
            try:
                pres = await _stage_async(req_id, "punc",
                                          _punc_sender(req_id, text, PRI_CLIP),
                                          deadline)
                if not pres.get("error"):
                    text = pres.get("text") or text
            except (asyncio.TimeoutError, TimeoutError):
                raise _timeout_error(req_id, name)
        wall = time.perf_counter() - t0
        _entry_stats["successes"] += 1
        _entry_stats["audio_total_s"] += audio_s
        _entry_stats["wall_total_s"] += wall
        _recent.appendleft({"id": job_id, "name": name[:40],
                            "audio_s": round(audio_s, 2),
                            "wall_s": round(wall, 2), "ok": True})
        return {"text": text, "job_id": job_id}
    except HTTPException:
        raise
    finally:
        _clear_pending(req_id)
        _finish_req(req_id)


def _cancel_all(req_id: str) -> None:
    """三队列置取消标记（worker 认领后跳过不白算）。redis 瞬断容忍——
    该请求由入口侧 WAIT_TIMEOUT 兜底。"""
    for qn in QUEUE_NAMES:
        try:
            _q[qn].cancel_add(req_id)
        except Exception:  # noqa: BLE001
            pass


def _timeout_error(req_id: str, name: str) -> HTTPException:
    """流水超时收口：三队列置取消标记（worker 认领后跳过不白算）+ 504。"""
    _entry_stats["timeouts"] += 1
    _recent_failures.appendleft(
        {"time": round(time.time(), 1), "status": 504,
         "reason": f"timeout >{WAIT_TIMEOUT_S:.0f}s ({name[:40]})"})
    _cancel_all(req_id)
    return HTTPException(status_code=504,
                         detail=f"timeout >{WAIT_TIMEOUT_S:.0f}s")


# ---------------- WS 流式（协议不变） ----------------

@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    """流式句级识别：二进制帧 PCM chunk 进，逐句 JSON 出。
    下行：{"text","begin","end","is_final":false}；结束 {"is_final":true}。"""
    if MODE == "batch":                             # 纯批量部署：没有流式服务
        await ws.accept()
        await ws.send_json({"error": "no stream service (MODE=batch)"})
        await ws.close(code=1013)
        return
    await ws.accept()
    sess = Session()
    loop = asyncio.get_running_loop()
    # 流式会话纳入 /queue /stats 观测（此前 WS 会话对运行视图完全不可见）
    sess.active_key = f"ws-{uuid.uuid4().hex[:8]}"
    with _active_lock:
        _active[sess.active_key] = {"id": None, "name": "ws",
                                    "audio_s": 0.0, "kind": "stream",
                                    "started_at": time.time()}
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            is_final = False
            if msg.get("bytes") is not None:
                try:
                    pcm = sess.ingest_bytes(msg["bytes"])
                except ValueError as e:
                    await ws.send_json({"error": str(e)})
                    continue
                if pcm is None or pcm.size == 0:
                    continue
                results = await loop.run_in_executor(
                    _ws_pool, sess.feed, pcm, False)
            elif msg.get("text"):
                try:
                    if not json.loads(msg["text"]).get("is_final"):
                        continue
                except Exception:  # noqa: BLE001
                    continue
                is_final = True
                results = await loop.run_in_executor(
                    _ws_pool, sess.feed, np.zeros(1, np.int16), True)
            else:
                continue
            # 逐句下发（二进制帧与 is_final 文本帧共用同一段循环）
            for s_ms, e_ms, text in results:
                await ws.send_json({"text": text, "begin": s_ms,
                                    "end": e_ms, "is_final": False})
            if is_final:
                await ws.send_json({"is_final": True})
                break
    except WebSocketDisconnect:
        pass
    finally:
        # 断连兜底：在途阶段取消（worker 跳过）+ 观测条目移除 + SHM 收口
        if sess.active_key is not None:
            with _active_lock:
                _active.pop(sess.active_key, None)
        rid = sess.inflight
        if rid is not None:
            _cancel_all(rid)
            for key in list(_pending):
                if key[0] == rid:
                    st = _pending.get(key)
                    if st is not None:
                        _complete_stage(st, {"text": "", "error": None,
                                             "raw": {}})
            _clear_pending(rid)
            _finish_req(rid)


# ---------------- 观测聚合（心跳文件跨 gateway 汇总） ----------------

_HB_CACHE_TTL_S = 1.0   # 心跳聚合缓存窗口：观测端点每秒最多扫盘一次
_hb_cache = {"t": 0.0, "data": []}


def _hb_cache_invalidate() -> None:
    """手动失效心跳缓存（测试写心跳文件后需立即可见的场合）。"""
    _hb_cache["t"] = 0.0


def _load_heartbeats() -> list:
    """新鲜心跳快照（worker-*.json，超龄跳过；1s TTL 缓存）——观测端点共用
    入口。此前 /health /metrics /stats /queue 每请求全目录重扫 + JSON 解析，
    /queue 一次触发 3 次；缓存后每秒至多一次。"""
    now = time.time()
    if now - _hb_cache["t"] < _HB_CACHE_TTL_S:
        return _hb_cache["data"]
    data = proc_common.iter_heartbeats(HEALTH_DIR, HB_STALE_S)
    _hb_cache["t"] = now
    _hb_cache["data"] = data
    return data


def _ready_state() -> tuple:
    """(ready, 明细)：期望角色表来自 supervisor 心跳，每个期望角色 ≥1 ready。"""
    hbs = _load_heartbeats()
    expected = {}
    for h in hbs:
        if h.get("role") == "supervisor":
            expected = h.get("expected") or {}
    counts = {role: 0 for role in expected}
    for h in hbs:
        r = h.get("role")
        if r in counts and h.get("ready"):
            counts[r] += 1
    ok = bool(expected) and all(v >= 1 for v in counts.values())
    return ok, {"expected": expected, "ready_counts": counts,
                "heartbeats": len(hbs)}


def _capacity_estimate() -> dict:
    """流式/批量容量估算（滚动样本 ≥10 条后用实测覆盖先验）。"""
    sent_s = clip_x = None
    try:
        client = ipc.shared_redis()
        sent = [float(x) for x in client.lrange("asr:sent_q", 0, -1)]
        clips = [x.decode() if isinstance(x, bytes) else x
                 for x in client.lrange("asr:clip_q", 0, -1)]
        if len(sent) >= 10:
            sent_s = sum(sent) / len(sent)
        if len(clips) >= 10:
            dts, auds = [], []
            for c in clips:
                try:
                    dt, au = c.split(",")
                    dts.append(float(dt))
                    auds.append(float(au))
                except ValueError:
                    continue
            if dts and sum(dts) > 0:
                clip_x = sum(auds) / sum(dts)
    except Exception:  # noqa: BLE001  观测失败回退先验
        pass
    workers = _expected_asr_workers()
    calibrated = sent_s is not None and clip_x is not None
    sent_s = sent_s or SENT_PRIOR_S
    clip_x = clip_x or BATCH_PRIOR_X
    asr_bound = workers / (sent_s * SENT_RATE_PER_ROUTE) if sent_s > 0 else 0
    vad_bound = VAD_ROUTES_PER_WORKER * max(1, _expected_vad_workers())
    return {"ws_capacity": round(min(vad_bound, asr_bound), 1),
            "batch_capacity": round(workers * clip_x, 1),
            "sent_prior_s": round(sent_s, 3), "clip_prior_x": round(clip_x, 1),
            "calibrated": calibrated}


def _expected_workers(role: str) -> int:
    for h in _load_heartbeats():
        if h.get("role") == "supervisor":
            return int((h.get("expected") or {}).get(role, 1))
    return 1


def _expected_asr_workers() -> int:
    return _expected_workers("asr")


def _expected_vad_workers() -> int:
    return _expected_workers("vad")


def _queue_wait_block(name: str) -> dict:
    """某队列 {depth, in/out/bad, wait: {n, avg/p50/p95/max}}——指标/队列视图共用。"""
    out = {"name": name, "depth": None, "in": None, "out": None, "bad": None,
           "avg_wait_ms": None, "wait": ipc.wait_stats(name)}
    try:
        out.update(_q[name].stats())
    except Exception:  # noqa: BLE001
        pass
    return out


@app.get("/health")
def health():
    """五键健康卡（容器 HEALTHCHECK / K8s 探针）：status/ready/version/shm/
    recent_failures。ready=false → 503。"""
    ok, detail = _ready_state()
    status = "ok" if ok else (
        "starting" if time.time() - _started_at < 600 else "degraded")
    body = {"status": status, "ready": ok, "version": IMAGE_VERSION,
            "shm": _shm_stats(),
            "recent_failures": list(_recent_failures)}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/metrics")
def metrics():
    """Prometheus 文本指标：请求计数 + **三队列等待时间**（avg/p50/p95/max）
    + 深度/吞吐 + worker 状态。"""
    if not METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="metrics disabled")
    hbs = _load_heartbeats()
    lines = []

    def m(name, value, labels=""):
        lines.append(f"{name}{labels} {value}")

    agg = collections.defaultdict(float)
    for h in hbs:
        if h.get("role") != "gateway":
            continue
        for k, v in (h.get("stats") or {}).items():
            if isinstance(v, (int, float)):
                agg[k] += v
    for k in ("requests", "successes", "timeouts", "rejections", "failed",
              "server_5xx", "sentences"):
        m(f"asr_e2e_{k}_total", int(agg.get(k, 0)))
    m("asr_e2e_audio_total_seconds", round(agg.get("audio_total_s", 0), 1))
    m("asr_e2e_wall_total_seconds", round(agg.get("wall_total_s", 0), 1))
    for name in QUEUE_NAMES:
        blk = _queue_wait_block(name)
        lbl = f'{{queue="{name}"}}'
        if blk["depth"] is not None:
            m("asr_queue_depth", blk["depth"], lbl)
            m("asr_queue_in_total", blk["in"], lbl)
            m("asr_queue_out_total", blk["out"], lbl)
            m("asr_queue_bad_total", blk["bad"], lbl)
            m("asr_queue_avg_wait_ms", blk["avg_wait_ms"], lbl)
        w = blk["wait"]
        if w["n"]:
            for q, key in (("avg", "avg_ms"), ("p50", "p50_ms"),
                           ("p95", "p95_ms"), ("max", "max_ms")):
                m("asr_queue_wait_ms", w[key],
                  f'{{queue="{name}",quantile="{q}"}}')
            m("asr_queue_wait_samples", w["n"], lbl)
    for h in hbs:
        role = h.get("role")
        wtag = h.get("wtag") or str(h.get("pid"))
        if role in ("asr", "vad", "punc"):
            m("asr_worker_ready", 1 if h.get("ready") else 0,
              f'{{worker="{wtag}",role="{role}"}}')
            m("asr_worker_stuck_seconds", float(h.get("stuck_s") or 0),
              f'{{worker="{wtag}",role="{role}"}}')
            npu = h.get("npu_mem") or {}
            if npu.get("allocated_mb") is not None:
                m("asr_worker_npu_mem_allocated_mb", npu["allocated_mb"],
                  f'{{worker="{wtag}"}}')
            st = h.get("stats") or {}
            m("asr_worker_jobs_total", int(st.get("jobs") or 0),
              f'{{worker="{wtag}",role="{role}"}}')
            m("asr_worker_errors_total", int(st.get("job_errors") or 0),
              f'{{worker="{wtag}",role="{role}"}}')
            if role == "asr":       # 凑批观测：平均批宽=tasks/claims，凑批率=批宽/配置宽
                m("asr_worker_batch_claims_total",
                  int(st.get("batches") or 0), f'{{worker="{wtag}"}}')
                m("asr_worker_batch_tasks_total",
                  int(st.get("batch_tasks") or 0), f'{{worker="{wtag}"}}')
                m("asr_worker_batch_max", int(st.get("batch_max") or 0),
                  f'{{worker="{wtag}"}}')
                for k, v in st.items():
                    if k.startswith("batchsz_"):
                        m("asr_worker_batch_size_total", int(v),
                          f'{{worker="{wtag}",bucket="{k[8:]}"}}')
        if role == "supervisor":
            for w, n in ((h.get("stats") or {}).get("crashes") or {}).items():
                m("asr_worker_crashes_total", int(n), f'{{worker="{w}"}}')
    shm = _shm_stats()
    if shm:
        m("asr_shm_used_mb", shm["used_mb"])
        m("asr_shm_total_mb", shm["total_mb"])
        m("asr_shm_pct", shm["pct"])
    ok, _ = _ready_state()
    m("asr_degraded", 0 if ok else 1)
    m('asr_build_info{version="%s"}' % IMAGE_VERSION, 1)
    return PlainTextResponse("\n".join(lines) + "\n",
                             media_type="text/plain; version=0.0.4")


# ---------------- 观测端点（admin 端口：/ /ready /stats /queue /promote /resources） ----------------

admin = FastAPI(title="ASR Admin", docs_url="/docs")


@admin.get("/")
def index():
    return {
        "service": "ASR Server",
        "endpoints": {
            "POST /v1/audio/transcriptions": "完整音频转写（multipart file 或 JSON audio_base64）",
            "WS  /ws": "流式句级识别（二进制 PCM/wav 帧 + 文本帧 is_final）",
            "GET /health": "健康卡（status/ready/version/shm/recent_failures）",
            "GET /metrics": "Prometheus 指标（含三队列等待时间 avg/p50/p95）",
            "GET /ready": "就绪探针（200/503）",
            "GET /stats": "累计统计 + RTF + 最近任务",
            "GET /queue": "三队列状态 + 等待时间 + 容量估算",
            "POST /queue/{job_id}/promote": "指定任务插队（clip 优先级 100→10）",
            "GET /resources": "主机资源 + 推理设备信息",
        },
    }


@admin.get("/ready")
def ready():
    ok, detail = _ready_state()
    return JSONResponse({"ready": ok, **detail},
                        status_code=200 if ok else 503)


@admin.get("/stats")
def stats():
    """累计统计（跨 gateway 心跳聚合，≤5s 延迟）+ RTF + 最近任务。"""
    hbs = _load_heartbeats()
    agg = collections.defaultdict(float)
    recent = []
    for h in hbs:
        if h.get("role") != "gateway":
            continue
        for k, v in (h.get("stats") or {}).items():
            if isinstance(v, (int, float)):
                agg[k] += v
        for r in (h.get("stats") or {}).get("recent") or []:
            recent.append(r)
    recent.sort(key=lambda r: r.get("time", 0), reverse=True)
    audio = agg.get("audio_total_s", 0.0)
    wall = agg.get("wall_total_s", 0.0)
    return {"processed": int(agg.get("successes", 0)),
            "failed": int(agg.get("failed", 0)),
            "requests": int(agg.get("requests", 0)),
            "timeouts": int(agg.get("timeouts", 0)),
            "sentences": int(agg.get("sentences", 0)),
            "audio_total_s": round(audio, 1),
            "wall_total_s": round(wall, 1),
            "rtf": round(wall / audio, 3) if audio > 0 else None,
            "realtime_speed": round(audio / wall, 2) if wall > 0 else None,
            "uptime_s": round(time.time() - _started_at, 1),
            "recent": recent[:10]}


@admin.get("/queue")
def queue_state():
    """三队列状态（深度/吞吐/**等待时间**）+ 各 gateway 在途请求 + 容量估算。"""
    out = {"queues": [_queue_wait_block(n) for n in QUEUE_NAMES],
           "running": [], "capacity": _capacity_estimate()}
    for h in _load_heartbeats():
        if h.get("role") != "gateway":
            continue
        out["running"].extend((h.get("stats") or {}).get("active") or [])
    return out


@admin.post("/queue/{job_id}/promote")
def promote(job_id: int):
    """任务插队：asr 队列中该 job 的 clip 优先级 100→10（仍让位于句子）。
    0 条命中 = 不在队列（在跑/已完成/未知）→ 404。"""
    try:
        n = _q["asr"].promote(str(job_id), PRI_PROMOTED)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    if not n:
        raise HTTPException(status_code=404,
                            detail="job not in queue (running/done/unknown)")
    return {"promoted": n, "job_id": job_id}


@admin.get("/resources")
def resources():
    """主机资源 + 推理设备信息（psutil + npu-smi，绝不在本进程碰 torch_npu）。"""
    import psutil
    devices = []
    try:
        out = __import__("subprocess").run(
            ["npu-smi", "info"], capture_output=True, text=True,
            timeout=5).stdout
        devices.append({"name": "Ascend NPU", "raw": out[-400:]})
    except Exception:  # noqa: BLE001
        pass
    mem = psutil.virtual_memory()
    return {"cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "mem_percent": mem.percent,
            "mem_used_gb": round(mem.used / 2**30, 1),
            "mem_total_gb": round(mem.total / 2**30, 1),
            "shm": _shm_stats(),
            "device": devices}


# ---------------- 心跳/清扫 ----------------

_started_at = time.time()


def _worker_snapshot() -> dict:
    """本进程实时状态快照（心跳落盘 + /health 聚合共用）。"""
    wtag = os.getenv("WORKER_TAG") or f"pid{os.getpid()}"
    with _active_lock:
        active = list(_active.values())[:16]
    stats = dict(_entry_stats)
    stats["recent"] = [{"time": round(time.time(), 1), **r}
                       for r in list(_recent)[:10]]
    stats["recent_failures"] = list(_recent_failures)
    stats["active"] = active
    if _result_q is not None:
        try:
            stats["mailboxes"] = ipc.mailbox_depths(
                {"gateway_queue": _result_q})
        except Exception:  # noqa: BLE001
            pass
    return {"wtag": wtag, "pid": os.getpid(), "role": "gateway",
            "device": None, "ready": True,   # 入口无模型：uvicorn 起来即可服务
            "uptime_s": round(time.time() - _started_at, 1),
            "warm": {}, "stats": stats, "ts": round(time.time(), 1)}


def _write_own_health() -> None:
    """心跳落盘（原子替换，避免读到半截 JSON）。"""
    try:
        proc_common.write_heartbeat(HEALTH_DIR, os.getpid(), _worker_snapshot())
    except Exception as exc:  # noqa: BLE001  心跳失败只影响监控，不影响服务
        log.warning("heartbeat write failed: %s", exc)


def _bg_loop() -> None:
    """后台线程：心跳落盘 + 孤儿 SHM 清扫 + 僵尸心跳文件清理。绝不能死。"""
    while True:
        try:
            _write_own_health()
            _sweep_orphan_shm()
            pct = _shm_used_pct()
            if pct >= 80:
                log.warning("/dev/shm 水位 %.0f%% —— 写满即 SIGBUS，加大 "
                            "--shm-size 或降低并发", pct)
            now = time.time()
            try:
                names = os.listdir(HEALTH_DIR)
            except OSError:
                names = []
            for name in names:
                if not name.startswith(("worker-", ".tmp-")):
                    continue
                path = os.path.join(HEALTH_DIR, name)
                try:
                    stale = now - os.path.getmtime(path) > HB_STALE_S * 2
                except OSError:
                    continue
                if stale:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
        except Exception:  # noqa: BLE001
            log.exception("heartbeat/sweep error (ignored)")
        time.sleep(5)


def gateway_main(i: int, port: int, admin_port: int, queues: dict,
                 result_q) -> None:
    """supervisor 经 mp spawn 启动网关进程：注入三队列句柄 → 绑定 →
    跑双 uvicorn server（业务 + admin）。SO_REUSEPORT 同端口多网关内核
    哈希分发；Windows 冒烟退化为独立端口。"""
    global _q, _result_q, _worker_origin, _main_loop
    global _ws_pool, _prep_pool
    _q = queues                      # {"vad", "asr", "punc"} 认领队列
    _result_q = result_q
    _worker_origin = int(i)
    os.environ["WORKER_TAG"] = f"gw-{i + 1}"
    _ws_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=128, thread_name_prefix="ws-feed")
    _prep_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=8, thread_name_prefix="prep")
    log.info("gateway-%d start (MODE=%s, queues=%s)", i, MODE,
             {n: _q[n].depth() for n in QUEUE_NAMES})

    threading.Thread(target=_bg_loop, daemon=True).start()
    threading.Thread(target=_result_pump, args=(result_q,), daemon=True).start()

    import uvicorn

    def _sock(p: int):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):   # Linux 生产：同端口多网关
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            bind_port = p
        else:                                  # Windows 冒烟：独立端口
            bind_port = p + 1000 + i
        s.bind(("0.0.0.0", bind_port))
        s.listen(2048)
        return s, bind_port

    biz_sock, biz_port = _sock(port)
    adm_sock, adm_port = _sock(admin_port)
    keep_alive = env_int("KEEP_ALIVE_TIMEOUT_S", 0, minimum=0)
    cfg_kwargs = dict(log_level=proc_common.log_level_name(),
                      timeout_keep_alive=keep_alive,
                      timeout_graceful_shutdown=4)
    srv_biz = uvicorn.Server(uvicorn.Config(app, **cfg_kwargs))
    srv_adm = uvicorn.Server(uvicorn.Config(admin, **cfg_kwargs))

    async def _serve():
        global _main_loop
        _main_loop = asyncio.get_running_loop()
        await asyncio.gather(srv_biz.serve(sockets=[biz_sock]),
                             srv_adm.serve(sockets=[adm_sock]))

    log.info("gateway-%d listening biz :%d admin :%d (reuseport hashed)",
             i, biz_port, adm_port)
    asyncio.run(_serve())


if __name__ == "__main__":
    raise SystemExit("gateway 由 supervisor.py 编排启动（注入队列句柄 + REUSEPORT "
                     "监听），不可直接运行。本地联调用 supervisor.py。")
