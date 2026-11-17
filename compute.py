"""worker 进程：vad / punc / asr 三类推理角色，各自独立队列消费。

拓扑（由 supervisor.py 编排启动，不可直接运行）：
    gateway → vad_queue  → vad-worker  （CPU：FSMN-VAD，整段切段 / 流式步进）
    gateway → asr_queue  → asr-worker  （NPU：SeACo-Paraformer，句子/clip）
    gateway → punc_queue → punc-worker （CPU：CT-Transformer，句尾/全文）
    三类 worker 结果统一回投 gateway_queues[origin]

设计要点（机制移植自 kokoro-tts-ascend 生产口径）：
- 每 worker 一个进程独占模型：funasr AutoModel 线程不安全被结构性消解；
- 认领都取 score 最低（句子/流式步进 0 < 插队 10 < 批量 100，与原统一池
  heapq 语义一致）；asr 角色**同桶宽认领**（一次取 ASR_MAX_BATCH_SIZE 条
  "差不多长度"的批前向，padding 有界），vad/punc 逐条；
- 心跳先行（加载期可见）+ BusyWatch 覆盖推理段：NPU aicore"不抛错永久
  阻塞"坏态由 supervisor 按 stuck_s 超时强杀重生；
- 心跳线程绝不碰 torch.npu（设备坏态副线程碰 NPU 会 SIGBUS 杀进程——
  kokoro 生产事故 20261109）：占用数据由主循环采样写缓存，心跳只读；
- 每任务推送**等待时间样本**（redis 滚动窗口）→ /metrics 展示各队列
  等待 avg/p50/p95；
- OOM 自愈（asr）：empty_cache + 整批重试一次，仍败降级逐条隔离（坏条只伤自家）；
- worker 死亡时在途任务由 supervisor 精确重派（重试一次，再失败弃置并
  补错误回 origin 网关——客户端立即 500 带原因）。
"""
import os
import re
import time

import numpy as np

import audio
import ipc
import proc_common
import protocol
from audio import SAMPLE_RATE
from proc_common import BusyWatch

log = proc_common.setup_logging("worker")

_stats = {"jobs": 0, "job_errors": 0, "job_drops": 0, "oom_retries": 0,
          "cancelled": 0, "batches": 0, "batch_tasks": 0, "batch_max": 0,
          "infer_total_s": 0.0, "audio_total_s": 0.0}
_busy = BusyWatch()
_warm = {"model": "loading"}     # loading / ok / fail
_state = {"device": None, "model": None, "role": None}
_npu_mem = {"allocated_mb": None, "reserved_mb": None}
_started_at = time.time()
ASR_BATCH_DEFAULT = 16   # 认领凑批宽默认值（与 proc_common.TUNABLE_PARAMS 同源，勿单改）

# 显存耗尽类异常识别（torch "out of memory"、独立缩写 OOM、昇腾 HBM
# 分配文案）——不能用裸子串 "oom"（会命中 room/zoom，误判触发无谓重试）。
_OOM_RE = re.compile(
    r"(?i)\bout of memory\b|\boom(?:[-_ ]?kill(?:ed)?)?\b"
    r"|hbm[^\n]{0,40}(?:alloc|memory)")


def _looks_like_oom(msg: str) -> bool:
    return bool(_OOM_RE.search(str(msg)))


def _batch_bucket(n: int) -> str:
    """批宽分档（凑批分布观测用）：1 / 2-4 / 5-8 / 9-16 / 17+。"""
    if n <= 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 8:
        return "5-8"
    if n <= 16:
        return "9-16"
    return "17+"


def _worker_snapshot() -> dict:
    """心跳快照（JSON 可序列化；stats_fn 由心跳线程周期调用——只读缓存，
    绝不在这里碰 torch.npu）。"""
    return {"role": _state["role"], "device": _state["device"],
            "ready": _warm.get("model") == "ok",
            "uptime_s": round(time.time() - _started_at, 1),
            "stuck_s": _busy.stuck_s(),
            "warm": dict(_warm), "npu_mem": dict(_npu_mem),
            "stats": {**dict(_stats), "ipc_dropped": protocol.drop_count()}}


def _sample_npu_mem() -> None:
    """NPU 显存占用采样（主循环每批后调用；非 NPU 空操作）。"""
    dev = _state["device"] or ""
    if not dev.startswith("npu"):
        return
    try:
        import torch
        idx = int(dev.split(":")[1]) if ":" in dev else 0
        _npu_mem["allocated_mb"] = round(
            torch.npu.memory_allocated(idx) / 2**20, 1)
        _npu_mem["reserved_mb"] = round(
            torch.npu.memory_reserved(idx) / 2**20, 1)
    except Exception:  # noqa: BLE001  观测失败不影响推理
        pass


def _read_task_audio(task: dict) -> np.ndarray:
    """读任务的 SHM PCM16 段 → float32（÷32767，与写入端同缩放）。

    SHM 连接/解码失败抛错——调用方捕获后单任务 error 回投（坏块只伤自家）。"""
    from multiprocessing import shared_memory

    piece = task["piece"]
    shm = shared_memory.SharedMemory(name=piece["shm_name"])
    try:
        return np.frombuffer(
            shm.buf[piece["offset"]:piece["offset"] + piece["samples"] * 2],
            dtype="<i2").astype(np.float32) / 32767.0
    finally:
        shm.close()  # unlink 归 gateway（创建侧/终态接收方）


def _push_capacity_sample(task: dict, infer_s: float) -> None:
    """容量估算滚动样本入 redis（LPUSH+LTRIM 50，跨 gateway 共享）：
    句子记占用秒；clip 记 (推理秒, 音频秒)。失败不影响推理。"""
    try:
        client = ipc.shared_redis()
        if task.get("kind") == protocol.KIND_SENT:
            client.lpush("asr:sent_q", f"{infer_s:.4f}")
            client.ltrim("asr:sent_q", 0, 49)
        else:
            audio_s = task["piece"]["samples"] / SAMPLE_RATE
            client.lpush("asr:clip_q", f"{infer_s:.4f},{audio_s:.4f}")
            client.ltrim("asr:clip_q", 0, 49)
    except Exception:  # noqa: BLE001  纯观测
        pass


def _cancelled(task: dict, claim_queue) -> bool:
    """任务被取消（超时/断连标记）：计数并返回 True，调用方跳过不白算。"""
    if claim_queue.is_cancelled(task.get("req_id", "")):
        _stats["cancelled"] += 1
        return True
    return False


def _reply(task: dict, result: dict, gateway_queues) -> None:
    """结果回投 origin 网关（成功/失败统一路径）。回投失败计 job_drops
    ——网关 WAIT_TIMEOUT 504 兜底（at-most-once 设计内）。"""
    qname = task.get("_qname", "asr")
    result.setdefault("timings", {})
    result["timings"].setdefault("q_wait", task.get("_qw_ms") or 0.0)
    result["timings"].setdefault("done_at", time.time())
    ok = protocol.put_result(
        gateway_queues[int(task.get("origin", 0))], result, log)
    if not ok:
        _stats["job_drops"] += 1


def _finish_task(task: dict, t0: float) -> None:
    """任务收尾簿记：等待时间样本 + 统计 +（asr）容量样本。"""
    _stats["jobs"] += 1
    infer_s = time.perf_counter() - t0
    _stats["infer_total_s"] += infer_s
    if task.get("_qw_ms") is not None:
        ipc.push_wait_sample(task.get("_qname", "asr"),
                             task["_qw_ms"])
    return infer_s


# ---------------- ASR worker（NPU） ----------------

def _empty_cache() -> None:
    """清缓存（npu/cuda 专属）。设备门禁在 import 之前——cpu/xpu 及 stub
    测试绝不 import torch（torch 的 aarch64 C 扩展在 QEMU 仿真下段错误）。"""
    dev = _state["device"] or ""
    if not dev.startswith(("npu", "cuda")):
        return
    try:
        import torch
        if dev.startswith("npu"):
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _infer_batch(waves: list, hotword) -> tuple:
    """**同桶凑批前向**：一次 generate 吃下整批（长度相近 padding 有界）。
    OOM 自愈：empty_cache + 整批重试一次；仍败/批路异常 → 降级逐条隔离。
    返回 (texts, errors) 与输入等长（对应位二选一为 None）。"""
    kw = {"disable_pbar": True, "batch_size": max(len(waves), 1)}
    if hotword:
        kw["hotword"] = hotword
    single_kw = {k: v for k, v in kw.items() if k != "batch_size"}

    def _batch():
        rs = _state["model"].generate(input=list(waves), **kw)
        if len(rs) != len(waves):
            raise RuntimeError(f"批结果数不符 {len(rs)} != {len(waves)}")
        return [r.get("text", "") for r in rs]

    try:
        return _batch(), [None] * len(waves)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if _looks_like_oom(msg):
            _stats["oom_retries"] += 1
            log.warning("OOM on batch(n=%d), empty_cache + retry once: %s",
                        len(waves), msg[:160])
            _empty_cache()
            try:
                return _batch(), [None] * len(waves)
            except Exception:  # noqa: BLE001  继续降级逐条
                pass
        else:
            log.warning("batch forward failed (n=%d), fall back to"
                        " sequential: %s", len(waves), msg[:160])
        texts, errs = [], []
        for w in waves:      # 逐条隔离：坏条只伤自家
            try:
                texts.append(_state["model"].generate(
                    input=w, **single_kw)[0].get("text", ""))
                errs.append(None)
            except Exception as e2:  # noqa: BLE001
                texts.append(None)
                errs.append(str(e2)[:200])
        return texts, errs


def _run_asr_jobs(claimed: list, gateway_queues, claim_queue) -> None:
    """一批 ASR 任务（同长度桶凑批）：取消过滤 + SHM 读取（逐条隔离）→
    按 hotword 分组批前向 → 逐条回投（批耗时按音频占比摊销进容量样本）。"""
    t0 = time.perf_counter()
    items = []                      # (task, wave)
    for _member, task in claimed:
        if _cancelled(task, claim_queue):
            continue
        try:
            items.append((task, _read_task_audio(task)))
        except Exception as exc:  # noqa: BLE001  坏块只伤自家
            _stats["jobs"] += 1
            _stats["job_errors"] += 1
            _reply(task, protocol.asr_result(
                task["req_id"], task["idx"],
                task.get("kind", protocol.KIND_CLIP),
                error=f"audio read failed: {str(exc)[:160]}"), gateway_queues)
    if not items:
        return
    # 按 hotword 分组（同请求热词一致；批间热词不同则分多桶前向）
    groups = {}
    for task, wave in items:
        groups.setdefault(task.get("hotword") or "", []).append((task, wave))
    _stats["batches"] += 1
    _stats["batch_tasks"] += len(items)
    if len(items) > _stats["batch_max"]:
        _stats["batch_max"] = len(items)
    _bsz = f"batchsz_{_batch_bucket(len(items))}"
    _stats[_bsz] = _stats.get(_bsz, 0) + 1
    infer_s = time.perf_counter() - t0
    for hw, pairs in groups.items():
        g_t0 = time.perf_counter()
        texts, errs = _infer_batch([w for _t, w in pairs], hw)
        g_s = time.perf_counter() - g_t0
        infer_s += g_s
        total_samples = sum(len(w) for _t, w in pairs) or 1
        for (task, wave), text, err in zip(pairs, texts, errs):
            _stats["jobs"] += 1
            if err is not None:
                _stats["job_errors"] += 1
                log.warning("req=%s idx=%s infer failed: %s",
                            task.get("req_id"), task.get("idx"), err[:160])
                _reply(task, protocol.asr_result(
                    task["req_id"], task["idx"],
                    task.get("kind", protocol.KIND_CLIP), error=err),
                    gateway_queues)
            else:
                _reply(task, protocol.asr_result(
                    task["req_id"], task["idx"],
                    task.get("kind", protocol.KIND_CLIP), text=text,
                    timings={"infer": g_s * 1000 * len(wave) / total_samples}),
                    gateway_queues)
                _push_capacity_sample(task, g_s * len(wave) / total_samples)
    log.info("batch n=%d infer=%.3fs audio=%.1fs",
             len(items), infer_s,
             sum(len(w) for _t, w in items) / SAMPLE_RATE)


def _warmup_asr(log, batch: int) -> None:
    """冒烟前向 + 最坏形状热身（触发 CANN 算子 JIT 编译；/root/.cache/ascend
    挂载持久化后只付一次代价）：单条 0.5s/30s + **认领宽凑批前向**（批路径
    JIT 对齐 ASR_MAX_BATCH_SIZE——不预热则首个真实请求付 JIT；顺带在启动期
    提前暴露批宽档的 OOM）。静音会跳过部分数值路径，用低幅噪声。"""
    rng = np.random.RandomState(0)
    for sec in (0.5, 30.0):
        wave = (rng.randn(int(SAMPLE_RATE * sec)) * 0.01).astype(np.float32)
        t0 = time.perf_counter()
        rs = _state["model"].generate(input=wave, disable_pbar=True)
        log.info("warmup %.1fs audio → %.2fs wall (%d chars)",
                 sec, time.perf_counter() - t0, len(rs[0].get("text", "")))
    # 批路径热身：认领宽条数、长度相近（同桶口径），覆盖 padded batch 前向的 JIT
    waves = [(rng.randn(int(SAMPLE_RATE * (2.0 + 0.03 * i))) * 0.01
              ).astype(np.float32) for i in range(batch)]
    t0 = time.perf_counter()
    rs = _state["model"].generate(input=waves, batch_size=batch,
                                  disable_pbar=True)
    log.info("warmup batch(%d) → %.2fs wall (%d results)",
             batch, time.perf_counter() - t0, len(rs))


# ---------------- VAD worker（CPU） ----------------

def _run_vad_batch(task: dict, gateway_queues) -> None:
    """vad_batch **边切边发**：整段音频按 30s 步进（流式 cache 状态机，与 WS
    同一机器、零边界误差），每步吐增量段（done=False）——gateway 收到即派发
    ASR clip，识别与切段重叠；最后 done=True 收尾。"""
    wave = _read_task_audio(task)
    cache = {}
    col = protocol.VadSegCollector()   # 事件→段状态机（与 gateway Session 共用实现）
    step = SAMPLE_RATE * 30
    total = len(wave)
    n_segs = 0
    t0 = time.perf_counter()
    final_segs = []
    if total == 0:
        pass
    else:
        for off in range(0, total, step):
            is_last = off + step >= total
            chunk = wave[off:off + step]
            res = _state["model"].generate(
                input=chunk, cache=cache, is_final=is_last, chunk_size=200,
                max_end_silence_time=800, disable_pbar=True)
            new = col.feed(res[0].get("value"))
            n_segs += len(new)
            if is_last:
                final_segs = new
            elif new:
                _reply(task, protocol.vad_result(
                    task["req_id"], task["idx"], new, done=False),
                    gateway_queues)      # 增量段：gateway 即刻派发 clip
    infer_s = _finish_task(task, t0)
    _reply(task, protocol.vad_result(
        task["req_id"], task["idx"], final_segs, done=True,
        timings={"infer": infer_s * 1000}), gateway_queues)
    log.info("req=%s idx=%s kind=vad_batch infer=%.3fs audio=%.1fs segs=%d"
             "（边切边发）", task.get("req_id"), task.get("idx"), infer_s,
             total / SAMPLE_RATE, n_segs)


def _run_vad_task(task: dict, gateway_queues, claim_queue) -> None:
    """单 VAD 任务：vad_batch=整段边切边发（见 _run_vad_batch）；vad_stream=
    流式步进（chunk+cache 随消息往返，worker 无状态，一步一结果）。"""
    if _cancelled(task, claim_queue):
        return
    kind = task.get("kind")
    if kind == protocol.KIND_VAD_BATCH:
        try:
            _run_vad_batch(task, gateway_queues)
        except Exception as exc:  # noqa: BLE001
            _finish_task(task, time.perf_counter())
            _stats["job_errors"] += 1
            _reply(task, protocol.vad_result(
                task["req_id"], task["idx"], [],
                error=str(exc)[:200]), gateway_queues)
        return
    t0 = time.perf_counter()
    try:
        chunk = np.asarray(task.get("chunk"), dtype=np.float32)
        cache = task.get("cache") or {}
        res = _state["model"].generate(
            input=chunk, cache=cache, is_final=bool(task.get("is_final")),
            chunk_size=task.get("chunk_size", 200),
            max_end_silence_time=task.get("max_end_silence_time", 800),
            disable_pbar=True)
        task["cache"] = cache          # mutated cache 回传（见 _reply）
        segs = [list(map(int, s)) for s in (res[0].get("value") or [])
                if len(s) == 2]
    except Exception as exc:  # noqa: BLE001
        _finish_task(task, t0)
        _stats["job_errors"] += 1
        _reply(task, protocol.vad_result(
            task["req_id"], task["idx"], [],
            error=str(exc)[:200]), gateway_queues)
        return
    infer_s = _finish_task(task, t0)
    result = protocol.vad_result(task["req_id"], task["idx"],
                                 segs, cache=task.get("cache"),
                                 timings={"infer": infer_s * 1000})
    _reply(task, result, gateway_queues)
    log.info("req=%s idx=%s kind=vad_stream infer=%.3fs segs=%d",
             task.get("req_id"), task.get("idx"), infer_s, len(segs))


# ---------------- PUNC worker（CPU） ----------------

def _run_punc_task(task: dict, gateway_queues, claim_queue) -> None:
    """单标点任务：文本进 → 补标点文本出。"""
    t0 = time.perf_counter()
    if _cancelled(task, claim_queue):
        return
    try:
        text = task.get("text") or ""
        out = text
        if text:
            out = _state["model"].generate(
                input=text, disable_pbar=True)[0].get("text", text)
    except Exception as exc:  # noqa: BLE001
        _finish_task(task, t0)
        _stats["job_errors"] += 1
        _reply(task, protocol.punc_result(
            task["req_id"], task["idx"], text,
            error=str(exc)[:200]), gateway_queues)
        return
    infer_s = _finish_task(task, t0)
    _reply(task, protocol.punc_result(
        task["req_id"], task["idx"], out,
        timings={"infer": infer_s * 1000}), gateway_queues)
    log.info("req=%s idx=%s infer=%.3fs chars=%d",
             task.get("req_id"), task.get("idx"), infer_s, len(text))


# ---------------- 加载与入口 ----------------

def _load_model(role: str, log) -> None:
    """设备解析（asr fail-fast）→ 模型加载 → 热身。显存动态分配不预占
    （随批形状伸缩，见 README）。任何失败直接退出（supervisor 指数退避
    重生；错误信息在 docker logs）。"""
    import asr_common
    _state["role"] = role
    if role == "asr":
        device = asr_common.resolve_device(os.getenv("DEVICE") or "auto")
        _state["device"] = device
        log.info("device resolved: %s", device)
        _state["model"] = asr_common.load_asr(device)
        _warmup_asr(log, proc_common.env_int(
            "ASR_MAX_BATCH_SIZE", ASR_BATCH_DEFAULT, minimum=1))
    elif role == "vad":
        _state["device"] = "cpu"
        _state["model"] = asr_common.load_vad("cpu")
        # 热身：一步流式 + 一次整段（两形态都过一遍）
        _state["model"].generate(input=np.zeros(3200, dtype=np.float32),
                                 cache={}, is_final=True, chunk_size=200,
                                 max_end_silence_time=800, disable_pbar=True)
        _state["model"].generate(input=np.zeros(16000, dtype=np.float32),
                                 disable_pbar=True)
    else:   # punc
        _state["device"] = "cpu"
        _state["model"] = asr_common.load_punc("cpu")
        _state["model"].generate(input="热身句子", disable_pbar=True)
    _warm["model"] = "ok"
    _sample_npu_mem()
    log.info("%s model ready (device=%s)", role, _state["device"])


def _claim_loop(role: str, idx: int, claim_queue, gateway_queues,
                run_tasks, claim_width: int = 1) -> None:
    """统一认领循环（三角色共用）：claim(claim_width) 取同长度桶一批 →
    BusyWatch 覆盖 → 执行 → 整批 ack。asr 用宽认领凑批前向；vad/punc 取 1。
    等待时间样本在 _finish_task 推送。"""
    tag = f"{role}-{idx}"
    worker_id = ipc.make_worker_id(role, idx, os.getpid())   # 死亡重派按它定位在途账本
    proc_common.start_heartbeat(tag=tag, pid=os.getpid(),
                                stats_fn=_worker_snapshot)
    _load_model(role, log)
    log.info("claim loop start (worker_id=%s, width=%d)", worker_id, claim_width)
    while True:
        try:
            claimed = claim_queue.claim(claim_width, worker_id)
        except Exception as exc:  # noqa: BLE001  redis 瞬断：等守护拉起
            log.warning("claim failed (redis restart window?): %s", exc)
            time.sleep(0.2)
            continue
        if not claimed:
            time.sleep(0.01)   # 队列空：轻轮询
            continue
        for _m, task in claimed:
            task["_qname"] = role   # 等待样本/回投日志的队列名
        _busy.enter()               # 看门狗：推理 hang 时心跳照常、stuck_s 增长
        try:
            run_tasks(claimed, gateway_queues, claim_queue)
        except Exception:  # noqa: BLE001  防御兜底（run_tasks 内部全捕获）
            log.exception("batch handling error (dropped, n=%d)", len(claimed))
            _stats["job_drops"] += len(claimed)
        finally:
            claim_queue.ack([m for m, _t in claimed], worker_id)  # 整批 ack
            _sample_npu_mem()
            _busy.exit()


def _run_vad_jobs(claimed: list, gateway_queues, claim_queue) -> None:
    """vad 单任务包装（认领宽度恒 1）。"""
    _run_vad_task(claimed[0][1], gateway_queues, claim_queue)


def _run_punc_jobs(claimed: list, gateway_queues, claim_queue) -> None:
    """punc 单任务包装（认领宽度恒 1）。"""
    _run_punc_task(claimed[0][1], gateway_queues, claim_queue)


def vad_main(idx: int, vad_queue, gateway_queues) -> None:
    """VAD worker 入口（CPU）：FSMN-VAD 整段边切边发 + 流式步进。"""
    _claim_loop("vad", idx, vad_queue, gateway_queues, _run_vad_jobs)


def punc_main(idx: int, punc_queue, gateway_queues) -> None:
    """PUNC worker 入口（CPU）：CT-Transformer 标点。"""
    _claim_loop("punc", idx, punc_queue, gateway_queues, _run_punc_jobs)


def asr_main(idx: int, asr_queue, gateway_queues) -> None:
    """ASR worker 入口（NPU）：SeACo-Paraformer **同桶凑批**前向。
    认领宽度 = ASR_MAX_BATCH_SIZE（一次取 N 个"差不多长度"的，一次 generate）。"""
    width = proc_common.env_int("ASR_MAX_BATCH_SIZE", ASR_BATCH_DEFAULT,
                                minimum=1)
    _claim_loop("asr", idx, asr_queue, gateway_queues, _run_asr_jobs,
                claim_width=width)


if __name__ == "__main__":
    raise SystemExit("compute 由 supervisor.py 编排启动（注入队列句柄），"
                     "不可直接运行。")
