#!/usr/bin/env python3
"""supervisor：多进程流水线的监督者（进程编排父进程，容器 PID 1）。

进程拓扑：
    redis-server    unix socket 队列服务端（纯内存、有界、无持久化）
    gateway ×E      HTTP 网关：接入/解码/流水编排/聚合（SO_REUSEPORT 内核哈希分发）
    vad-worker ×V   FSMN-VAD（CPU：整段切段 + 流式步进，独立队列）
    punc-worker ×P  CT-Transformer（CPU：标点，独立队列）
    asr-worker ×N   SeACo-Paraformer（NPU：句子/clip，独立队列）

兜底（机制移植自 kokoro-tts-ascend 生产验证口径）：
- 任一子进程退出 → 指数退避后同角色拉起（稳定 60s 清零）；
- 心跳停更（HEARTBEAT_STALE_S）或 busy 卡死（stuck_s > STUCK_KILL_S）→ 强杀重启；
- 死亡立即精确重派在途任务（requeue_worker，重试一次；再失败弃置并补
  asr_result 错误回 origin 网关——客户端立即 500 带原因而非等 504）；
- 全部角色就绪后连续 WEDGE_RESTART_S 零成功且仍有失败 → exit(1) 交容器重启；
- NPU worker 串行启动（上一个 ready 再拉下一个，并行 init 抢卡曾放大坏态雪崩）；
- SIGTERM 全组退出。

**绝不 import torch_npu**：父进程一旦创建 CANN 上下文就占一份显存挤占
worker 预算；显存探测走短命子进程（kokoro 生产教训）。

env：GATEWAY_WORKERS / ASR_WORKERS / PORT / ADMIN_PORT / ASR_HEALTH_DIR 等，
全量见 proc_common.TUNABLE_PARAMS（启动时逐项打印）。
"""
import json
import multiprocessing as mp
import os
import signal
import sys
import time

import ipc
import proc_common
import protocol

log = proc_common.setup_logging("supervisor")

GATEWAY_WORKERS_N = proc_common.env_int("GATEWAY_WORKERS", 2, minimum=1)
ASR_N = proc_common.env_int("ASR_WORKERS", 8, minimum=1)
VAD_N = proc_common.env_int("VAD_WORKERS", 2, minimum=1)
PUNC_N = proc_common.env_int("PUNC_WORKERS", 2, minimum=1)
PORT = proc_common.env_int("PORT", 10095, minimum=1)
ADMIN_PORT = proc_common.env_int("ADMIN_PORT", 10096, minimum=1)
HEALTH_DIR = os.getenv("ASR_HEALTH_DIR", "/tmp/asr-health")
MAX_PENDING_REQUESTS = proc_common.env_int("MAX_PENDING_REQUESTS", 8, minimum=1)
MAX_INFLIGHT_TASKS = proc_common.env_int("MAX_INFLIGHT_TASKS", 512, minimum=1)
WORKER_MAILBOX_SIZE = proc_common.env_int("WORKER_MAILBOX_SIZE", 1024, minimum=1)
HEARTBEAT_STALE_S = proc_common.env_float("HEARTBEAT_STALE_S", 30.0, minimum=15.0)
STUCK_KILL_S = proc_common.env_float("STUCK_KILL_S", 300.0, minimum=30.0)
SPAWN_READY_TIMEOUT_S = proc_common.env_float("SPAWN_READY_TIMEOUT_S", 360.0, minimum=0.0)
WEDGE_RESTART_S = proc_common.env_float("WEDGE_RESTART_S", 900.0, minimum=0.0)
# 毒块防护：同一任务累计重派超过该次数 → 放弃（补 asr_result 错误，请求 500）
REDISPATCH_MAX = proc_common.env_int("REDISPATCH_MAX", 1, minimum=0)
# TTL 兜底线（2×看门狗击杀线）：死亡重派走秒级路径，此值只兜 supervisor 漏检
SWEEP_TTL_S = 2 * STUCK_KILL_S

ctx = mp.get_context("spawn")
procs = {}        # (role, i) → Process
_SERIAL_ROLES = {"asr"}   # NPU worker 同类串行启动（CPU vad/punc/gateway 并行秒起）
starting = {}     # (role, i) → 拉起时刻（串行启动闸）
mailboxes = {}    # 名字 → 队列对象（Redis 键名元数据；心跳深度探针用）
_spawn_ts = {}    # (role, i) → 最近拉起时刻（稳定判定）
_fail_count = {}  # (role, i) → 连续崩溃次数（指数退避）
_next_ok = {}     # (role, i) → 最早允许重启时刻
_crash_total = {}  # (role, i) → 累计崩溃次数

EXPECTED_ROLES = {"gateway": GATEWAY_WORKERS_N, "vad": VAD_N,
                  "punc": PUNC_N, "asr": ASR_N}

_redis_proc = None        # redis-server 子进程（main 拉起、_check_redis 守护）
_task_queues = {}         # 认领队列实例 {"vad","asr","punc"}（sweeper/requeue 用）
_redis_client = None      # sweeper 用 redis 客户端（redis 重启时重建）
_sweep_totals = {"redispatched": 0, "gaveup": 0}
_bad_seen = {"n": None}


def _wtag(role: str, i: int) -> str:
    """worker 心跳/日志标签（单一映射）：gateway 用 gw-N（1 起），asr 用 asr-N。
    与 gateway._worker_snapshot / compute 的 wtag 必须同源。"""
    return f"gw-{i + 1}" if role == "gateway" else f"{role}-{i}"


def _as_int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def spawn_one(role: str, i: int) -> None:
    """拉起角色进程；各角色只接收自己消费/投递的 mailbox 句柄（最小权限）。"""
    gqs = mailboxes["gateway_queues"]   # 全部网关邮箱（按 origin 回投需要完整列表）
    if role == "gateway":
        import gateway
        target = gateway.gateway_main
        qmap = {"vad": mailboxes["vad_queue"], "asr": mailboxes["asr_queue"],
                "punc": mailboxes["punc_queue"]}
        args = (i, PORT, ADMIN_PORT, qmap, gqs[i])
    else:  # vad / punc / asr
        import compute
        target = {"vad": compute.vad_main, "punc": compute.punc_main,
                  "asr": compute.asr_main}[role]
        args = (i, mailboxes[f"{role}_queue"], gqs)
    tag = _wtag(role, i)
    p = ctx.Process(target=target, args=args, name=tag)
    # 先登记后 start：信号处理器在主线程字节码间隙执行，start 与登记之间
    # 被打断的话 shutdown 遍历不到该子进程 → 孤儿持有 NPU 显存/监听端口。
    procs[(role, i)] = p
    p.start()
    _spawn_ts[(role, i)] = time.time()
    _crash_total.setdefault((role, i), 0)   # 预置：心跳线程遍历期间不得增键
    if SPAWN_READY_TIMEOUT_S and role in _SERIAL_ROLES:
        starting[(role, i)] = time.time()
    log.info("spawned [%s] pid=%s", tag, p.pid)


def shutdown(signum, frame):
    log.info("exit signal received, shutting down all children")
    for p in procs.values():
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    for p in procs.values():   # 限时等待，避免 in-flight 响应被硬切
        try:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)
        except Exception:  # noqa: BLE001
            pass
    if _redis_proc is not None:
        try:
            _redis_proc.terminate()
            _redis_proc.join(timeout=2)
        except Exception:  # noqa: BLE001
            pass
    _cleanup_own_heartbeat()
    sys.exit(0)


def _cleanup_own_heartbeat() -> None:
    try:
        os.unlink(os.path.join(HEALTH_DIR, f"worker-{os.getpid()}.json"))
    except OSError:
        pass


def _clean_stale_heartbeats() -> None:
    """启动时无条件清掉上一代进程遗留的心跳文件（docker restart 后 PID
    复用撞旧快照会让串行启动闸提前放行/reap 误杀）。此刻本代 worker 未起，全删安全。"""
    try:
        for name in os.listdir(HEALTH_DIR):
            if name.startswith(("worker-", ".tmp-")):
                try:
                    os.unlink(os.path.join(HEALTH_DIR, name))
                except OSError:
                    continue
    except OSError:
        pass


def _worker_ready(pid: int) -> bool:
    """心跳文件里该 pid 是否已 ready（读不了按未就绪，下轮再查）。"""
    try:
        with open(os.path.join(HEALTH_DIR, f"worker-{pid}.json"),
                  encoding="utf-8") as f:
            return bool(json.load(f).get("ready"))
    except (OSError, ValueError):
        return False


def _prune_starting(now: float) -> None:
    """放行启动闸：worker ready / 死亡 / 超时——同角色下一个可拉起。"""
    for key in list(starting):
        p = procs.get(key)
        if p is None or not p.is_alive():
            starting.pop(key, None)
        elif _worker_ready(p.pid):
            log.info("[%s-%d] started (ready in %.0fs)",
                     key[0], key[1], now - starting[key])
            starting.pop(key, None)
        elif SPAWN_READY_TIMEOUT_S and now - starting[key] > SPAWN_READY_TIMEOUT_S:
            log.warning("[%s-%d] not ready after %.0fs, releasing start gate",
                        key[0], key[1], now - starting[key])
            starting.pop(key, None)


def _role_start_busy(role: str) -> bool:
    return any(k[0] == role for k in starting)


def _wait_started(key) -> None:
    """冷启动串行：阻塞等刚拉起的 NPU worker ready/死亡/超时。
    等待期间仍跑看门狗/Redis 守护（hung NPU init 的 stuck_s 不能没人读）。"""
    if not SPAWN_READY_TIMEOUT_S:
        return
    deadline = time.time() + SPAWN_READY_TIMEOUT_S
    while time.time() < deadline:
        try:
            _check_redis()
        except Exception:  # noqa: BLE001
            log.exception("redis guardian failed during startup (continue)")
        try:
            reap_hung()
        except Exception:  # noqa: BLE001
            log.exception("reap_hung failed during startup (continue)")
        _prune_starting(time.time())
        if key not in starting:
            return
        time.sleep(0.5)


def _shm_total_bytes() -> int:
    u = proc_common.shm_usage()
    return -1 if u is None else u["total"]


def check_shm() -> None:
    """启动自检：/dev/shm 过小 → 音频 SHM 块写满 tmpfs 触发 SIGBUS。"""
    total = _shm_total_bytes()
    if total < 0:
        log.warning("/dev/shm 容量探测失败（跳过 SHM 自检；Windows 开发机正常）")
        return
    if total < 2 * 2**30:
        log.error("/dev/shm 只有 %.0fMB（要求 ≥2g）—— 音频 SHM 写满会触发"
                  " SIGBUS。修复：docker run 加 --shm-size 2g 并重启容器",
                  total / 2**20)
    else:
        log.info("/dev/shm %.1fGB (SHM 自检通过)", total / 2**30)


def classify_heartbeat(path: str, now: float) -> tuple:
    """心跳判决：(verdict, detail)，verdict ∈ {"ok", "stale", "stuck"}。

    stale = 文件停更 ≥HEARTBEAT_STALE_S（心跳线程死亡/整机卡死）；
    stuck = 心跳新鲜但 busy 卡住 >STUCK_KILL_S（模型推理 hang 时心跳照常
    落盘，光看 mtime 是盲区，必须读快照里的 stuck_s）。"""
    try:
        if now - os.path.getmtime(path) > HEARTBEAT_STALE_S:
            return "stale", f"heartbeat stale >= {HEARTBEAT_STALE_S}s"
    except OSError:
        return "ok", ""
    try:
        with open(path, encoding="utf-8") as f:
            snap = json.load(f)
    except (OSError, ValueError):
        return "ok", ""
    try:
        stuck = float(snap.get("stuck_s") or 0)
    except (TypeError, ValueError):
        return "ok", ""
    if stuck > STUCK_KILL_S:
        return "stuck", f"busy stuck {stuck:.0f}s > {STUCK_KILL_S:.0f}s (inference hang)"
    return "ok", ""


def reap_hung() -> None:
    """hang 清扫：心跳停更（stale）或推理卡死（stuck）→ 强杀。仅杀自己的子进程。"""
    now = time.time()
    for path in proc_common.heartbeat_files(HEALTH_DIR):
        pid_str = os.path.basename(path)[len("worker-"):-5]
        if not pid_str.isdigit():
            continue
        verdict, detail = classify_heartbeat(path, now)
        if verdict == "ok":
            continue
        pid = int(pid_str)
        for key, p in procs.items():
            if p.pid == pid:
                log.warning("[%s-%s] %s, judged hung, force restart",
                            key[0], key[1], detail)
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass
                break


_wedge_state = {"last_res": None, "last_res_ts": 0.0,
                "last_tmo": None, "last_tmo_ts": 0.0,
                "last_5xx": None, "last_5xx_ts": 0.0}


def _on_giveup(chunk: dict) -> None:
    """重派超限/坏 member 的放弃收口：按任务类型补一条错误结果回 origin 网关
    ——客户端立即 500 带原因（而不是等 WAIT_TIMEOUT 504）。"""
    try:
        origin = int(chunk.get("origin", 0))
        qs = mailboxes.get("gateway_queues") or []
        if origin < 0 or origin >= len(qs):
            return
        err = "task dropped: worker died twice (retry exhausted)"
        t = chunk.get("type")
        if t == protocol.VAD_TASK:
            msg = protocol.vad_result(chunk.get("req_id"), chunk.get("idx", 0),
                                      [], error=err)
        elif t == protocol.PUNC_TASK:
            msg = protocol.punc_result(chunk.get("req_id"), chunk.get("idx", 0),
                                       None, error=err)
        else:
            msg = protocol.asr_result(chunk.get("req_id"), chunk.get("idx", 0),
                                      chunk.get("kind", protocol.KIND_CLIP),
                                      error=err)
        protocol.put_result(qs[origin], msg)
    except Exception:  # noqa: BLE001  通知失败不影响清扫
        pass


def _watch_queue_bad() -> None:
    """坏 member 增量告警：每轮汇总三条认领队列的累计计数，增量出现即
    WARNING（滚动升级混跑/载荷损坏唯一的主动信号）。"""
    if not _task_queues:
        return
    try:
        total = sum(int((q.stats() or {}).get("bad") or 0)
                    for q in _task_queues.values())
    except Exception:  # noqa: BLE001
        return
    prev = _bad_seen["n"]
    if prev is not None and total > prev:
        log.warning("queue bad members discarded: +%d (total %d) — "
                    "payload corruption or version-mixed members",
                    total - prev, total)
    _bad_seen["n"] = total


def _sweep_inflight() -> None:
    """在途重派扫描：per-worker 在途账本按 worker 死亡精确重派（重派条件是
    「worker 已死」而非「TTL 超时」——慢 worker 永不被误重派）。"""
    global _redis_client
    if _redis_client is None:
        _redis_client = ipc._make_redis_client()
    if not _task_queues:
        return
    try:
        alive = set()
        for p in procs.values():
            if not p.pid:
                continue
            try:
                if p.is_alive():
                    alive.add(p.pid)
            except Exception:  # noqa: BLE001
                continue
        r = {"redispatched": 0, "gaveup": 0}
        for q in _task_queues.values():
            r2 = q.sweep(alive, time.time(), SWEEP_TTL_S, REDISPATCH_MAX,
                         on_giveup=_on_giveup)
            r["redispatched"] += r2["redispatched"]
            r["gaveup"] += r2["gaveup"]
    except Exception as exc:  # noqa: BLE001  redis 瞬断：下轮再试
        log.warning("inflight sweep failed (retry next cycle): %s", exc)
        return
    _sweep_totals["redispatched"] += r["redispatched"]
    _sweep_totals["gaveup"] += r["gaveup"]
    if r["redispatched"]:
        log.warning("inflight redispatched %d (dead worker precise requeue;"
                    " TTL 兜底=%.0fs) total=%d gaveup=%d",
                    r["redispatched"], SWEEP_TTL_S,
                    _sweep_totals["redispatched"], _sweep_totals["gaveup"])
    if r["gaveup"]:
        log.error("inflight discarded %d tasks (bad member / redispatch over"
                  " limit %d; affected requests → 500 via asr_result)",
                  r["gaveup"], REDISPATCH_MAX)


def _check_redis() -> None:
    """redis-server 守护：死亡 = 队列与在途一起丢失（服务端内存态，设计如此），
    损失有界：受影响请求走 WAIT_TIMEOUT 504 兜底。自动重启，队列从零起。"""
    global _redis_proc, _redis_client
    if _redis_proc is None or _redis_proc.poll() is None:
        return
    log.critical("redis-server died (rc=%s), restarting (queues+in-flight "
                 "lost, requests fall back to WAIT 504)", _redis_proc.returncode)
    _redis_proc = ipc.start_redis_server()
    _redis_client = None


def _gateway_activity(now: float) -> tuple:
    """聚合网关心跳：(requests, successes, timeouts, server_5xx, role_ready)。
    文件过旧/损坏/字段异常一律容错（按 0）；无网关心跳返回 None。"""
    req = res = tmo = s5 = 0
    n_gw = 0
    role_ready = {r: 0 for r in EXPECTED_ROLES}
    for d in proc_common.iter_heartbeats(HEALTH_DIR, HEARTBEAT_STALE_S):
        role = d.get("role")
        if d.get("ready") and role in role_ready:
            role_ready[role] += 1
        if role != "gateway" or not d.get("ready"):
            continue
        n_gw += 1
        s = d.get("stats") or {}
        req += _as_int(s.get("requests"))
        res += _as_int(s.get("successes"))
        tmo += _as_int(s.get("timeouts"))
        s5 += _as_int(s.get("server_5xx"))
    if n_gw == 0:
        return None
    return req, res, tmo, s5, role_ready


def check_wedge(now: float) -> None:
    """端到端假死兜底：全部角色就绪后，连续 WEDGE_RESTART_S 零成功、且近期
    仍有超时/5xx → os._exit(1) 交容器重启。用请求级计数判定（不依赖任何
    传输的队列内部状态）：覆盖 redis 死亡/上游生产卡死等"请求进得去、
    结果出不来"的服务级假死。"""
    if WEDGE_RESTART_S <= 0:
        return
    act = _gateway_activity(now)
    if act is None:
        _wedge_state.update(last_res=None, last_tmo=None, last_5xx=None)
        return
    req, res, tmo, s5, role_ready = act
    if any(role_ready.get(r, 0) < 1 for r in EXPECTED_ROLES):
        _wedge_state.update(last_res=None, last_tmo=None, last_5xx=None)
        return
    st = _wedge_state
    if st["last_res"] is None:
        st.update(last_res=res, last_res_ts=now, last_tmo=tmo,
                  last_5xx=s5, last_5xx_ts=now)
        return
    # 计数变化即刷新时间戳（成员变化/恢复期不误判）
    if res != st["last_res"]:
        st.update(last_res=res, last_res_ts=now)
    if tmo != st["last_tmo"]:
        st.update(last_tmo=tmo, last_tmo_ts=now)
    if s5 != st["last_5xx"]:
        st.update(last_5xx=s5, last_5xx_ts=now)
    quiet = now - st["last_res_ts"]
    recent_timeout = now - st["last_tmo_ts"] <= WEDGE_RESTART_S / 2
    recent_5xx = s5 > 0 and now - st["last_5xx_ts"] <= WEDGE_RESTART_S / 2
    if quiet >= WEDGE_RESTART_S and (recent_timeout or recent_5xx) and req > 0:
        log.critical("end-to-end wedge: no successful result for %.0fs "
                     "(requests=%d timeouts=%d server_5xx=%d, all roles ready)"
                     " — exiting for container restart", quiet, req, tmo, s5)
        for h in log.handlers:
            try:
                h.flush()
            except Exception:  # noqa: BLE001
                pass
        os._exit(1)


def supervisor_snapshot() -> dict:
    return {"role": "supervisor", "ready": True,
            "uptime_s": round(time.time() - _started_at, 1),
            "warm": {},
            "stats": {"crashes": {_wtag(r, i): n for (r, i), n in
                                  list(_crash_total.items())}},
            "expected": dict(EXPECTED_ROLES),
            "mailboxes": ipc.mailbox_depths(mailboxes)}


_started_at = time.time()


def _safe_step(name: str, fn, *args):
    """监控子步骤独立兜底：一项失败不能连带跳过其余项。"""
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        log.exception("monitor step %s failed (continue)", name)
        return None


def monitor_once() -> None:
    """单轮监控：redis 守护 + hang 清扫 + 假死判死 + 退出重启（指数退避）。"""
    _safe_step("redis", _check_redis)
    _safe_step("sweep_inflight", _sweep_inflight)
    _safe_step("queue_bad", _watch_queue_bad)
    _safe_step("reap_hung", reap_hung)
    now = time.time()
    _safe_step("wedge", check_wedge, now)
    _safe_step("prune_starting", _prune_starting, now)
    for (role, i), p in list(procs.items()):
        try:
            alive = p.is_alive()
        except Exception:  # noqa: BLE001
            alive = False
        if alive:
            if now - _spawn_ts.get((role, i), now) > 60:
                _fail_count[(role, i)] = 0   # 稳定运行满 60s：退避清零
            continue
        # 死亡收口立即执行（不能等串行闸/退避）：删心跳文件 + 在途精确重派
        pid = p.pid
        try:
            os.unlink(os.path.join(HEALTH_DIR, f"worker-{pid}.json"))
        except OSError:
            pass
        if _task_queues:
            for q in _task_queues.values():
                try:
                    n = q.requeue_worker(
                        ipc.make_worker_id(role, i, pid),
                        max_redispatch=REDISPATCH_MAX, on_giveup=_on_giveup)
                    if n:
                        log.warning("[%s-%d] requeued %d in-flight from dead"
                                    " worker (over-limit tasks dropped → 500)",
                                    role, i, n)
                except Exception:  # noqa: BLE001  redis 瞬断：sweep 兜底
                    pass
        # 再生决策（串行启动闸/指数退避）
        if role in _SERIAL_ROLES and _role_start_busy(role):
            continue
        n = _fail_count.get((role, i), 0)
        if now < _next_ok.get((role, i), 0):
            continue   # 退避等待期
        delay = min(30.0, 0.5 * (2 ** n))
        _fail_count[(role, i)] = n + 1
        _crash_total[(role, i)] = _crash_total.get((role, i), 0) + 1
        _next_ok[(role, i)] = now + delay
        rc = p.exitcode if isinstance(p.exitcode, int) else -1
        log.warning("[%s-%d] exited(rc=%s), respawn in %.1fs (consecutive #%d)",
                    role, i, rc, delay, n + 1)
        procs.pop((role, i), None)
        try:
            spawn_one(role, i)
        except Exception as exc:  # noqa: BLE001  拉起失败不杀 supervisor 自身
            log.error("[%s-%d] respawn failed: %s", role, i, exc)
            procs.setdefault((role, i), p)   # 槽位放回，后续循环仍会重试


def _monitor_step() -> None:
    """主循环单步兜底：monitor_once 残余异常不能杀死 supervisor 主线程。"""
    try:
        monitor_once()
    except Exception:  # noqa: BLE001
        log.exception("monitor cycle failed (supervisor kept alive)")


def create_mailboxes() -> None:
    """Mailbox 创建（全部有界，背压）：下游满 → 上游停 → 入口拒 429。
    三条认领队列（vad/asr/punc）优先级语义同构：流式/句尾 0 < 插队 10 < 批量 100。"""
    for idx, name in enumerate(("vad", "asr", "punc")):
        mailboxes[f"{name}_queue"] = ipc.AsrClaimQueue(
            MAX_INFLIGHT_TASKS, index=idx, name=name)
    mailboxes["gateway_queues"] = [
        ipc.InstrumentedQueue(WORKER_MAILBOX_SIZE, index=3 + i,
                              name=f"gateway_queue_{i}")
        for i in range(GATEWAY_WORKERS_N)]          # 终态回投（按 origin 路由）


def spawn_all() -> None:
    """按角色配置拉起全部子进程：gateway/vad/punc 并行（CPU 秒起）→
    asr-0 阻塞等 ready → 其余 asr 串行（NPU init 不抢卡）。"""
    for i in range(GATEWAY_WORKERS_N):
        spawn_one("gateway", i)
    for i in range(VAD_N):
        spawn_one("vad", i)
    for i in range(PUNC_N):
        spawn_one("punc", i)
    spawn_one("asr", 0)
    _wait_started(("asr", 0))
    for i in range(1, ASR_N):
        spawn_one("asr", i)
        _wait_started(("asr", i))


def _capacity_check() -> None:
    """启动容量自检（短命子进程探 NPU 容量——父进程绝不 import torch_npu）。
    ASR_N × (权重 ~1.3GB + 激活余量 0.3GB×批宽，按 ASR_MAX_BATCH_SIZE 最坏
    形状=同批全 30s 长_clip 口径) 超 95% 卡容量记 ERROR 但不拒绝启动
    （vNPU 共享卡由编排层决定）。非 NPU 环境静默跳过。"""
    if (os.getenv("DEVICE") or "auto") == "cpu":
        return
    try:
        import subprocess
        out = subprocess.run(
            [sys.executable, "-c",
             "import torch, torch_npu;"
             "print(torch.npu.get_device_properties(0).total_memory // 2**20)"],
            capture_output=True, text=True, timeout=60).stdout.strip()
        total_mb = int(out.splitlines()[-1])
    except Exception:  # noqa: BLE001  无卡/无 torch_npu：跳过
        return
    batch = proc_common.env_int("ASR_MAX_BATCH_SIZE", 16, minimum=1)
    per_mb = 1300 + 300 * batch   # 权重 + 激活/编译缓存余量（batch4 档实测 1.2GB）
    phys_mb = ASR_N * per_mb
    if phys_mb > total_mb * 0.95:
        log.error("NPU capacity check: asr x%d (batch%d, %.1fGB/worker) ≈ "
                  "%.1fGB > 95%% of %.1fGB — reduce ASR_WORKERS or "
                  "ASR_MAX_BATCH_SIZE", ASR_N, batch, per_mb / 1024,
                  phys_mb / 1024, total_mb / 1024)
    else:
        log.info("NPU capacity check: asr x%d (batch%d) ≈ %.1fGB / %.1fGB "
                 "(%.1f%%)", ASR_N, batch, phys_mb / 1024, total_mb / 1024,
                 100.0 * phys_mb / max(total_mb, 1))


def main() -> None:
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    proc_common.dump_config()          # 启动第一件事：环境变量自证（print 直写，不受 LOG_LEVEL 影响）
    check_shm()
    _clean_stale_heartbeats()
    create_mailboxes()
    global _redis_proc, _task_queues
    _redis_proc = ipc.start_redis_server()
    _task_queues = {"vad": mailboxes["vad_queue"],
                    "asr": mailboxes["asr_queue"],
                    "punc": mailboxes["punc_queue"]}
    if _redis_proc is None:
        log.info("external redis ready (url=%s)", ipc.IPC_REDIS_URL)
    else:
        log.info("redis-server ready (unix socket=%s, pid=%s)",
                 ipc.IPC_REDIS_SOCKET, _redis_proc.pid)
    proc_common.start_heartbeat(tag="supervisor", pid=os.getpid(),
                                stats_fn=supervisor_snapshot)
    _capacity_check()
    spawn_all()
    log.info("orchestration complete gateway x%d + vad x%d + punc x%d"
             " + asr x%d, listening :%d (reuseport hashed dispatch)",
             GATEWAY_WORKERS_N, VAD_N, PUNC_N, ASR_N, PORT)
    while True:
        time.sleep(2)
        _monitor_step()


if __name__ == "__main__":
    main()
