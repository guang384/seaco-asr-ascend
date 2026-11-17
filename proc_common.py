"""进程公共件：env 解析 + 日志 + BusyWatch 看门狗 + 心跳落盘。

被 compute（ASR worker）/ gateway / supervisor 复用。跨进程消息传输
（Redis 队列）在 ipc.py——本模块不依赖 redis，纯本进程内机制。
"""
import json
import logging
import os
import time

HEALTH_DIR = os.getenv("ASR_HEALTH_DIR", "/tmp/asr-health")

# 可调配参数清单（名 → 默认值）：supervisor 启动时逐项打印实际值。
# 声明于此 = 参数面单一事实来源；新增参数必须登记，否则启动日志不显示。
TUNABLE_PARAMS = [
    # ── 进程池数 ──
    ("GATEWAY_WORKERS", "2"), ("ASR_WORKERS", "8"),
    ("VAD_WORKERS", "2"), ("PUNC_WORKERS", "2"),
    # ── 凑批（同长度桶一次取 N 个批前向）──
    ("ASR_MAX_BATCH_SIZE", "16"),
    # ── 背压闸门 ──
    ("MAX_PENDING_REQUESTS", "8"), ("MAX_INFLIGHT_TASKS", "512"),
    ("REDISPATCH_MAX", "1"), ("WORKER_MAILBOX_SIZE", "1024"),
    # ── 超时与对齐 ──
    ("WAIT_TIMEOUT_S", "120"),
    ("HEARTBEAT_STALE_S", "30"), ("STUCK_KILL_S", "300"),
    ("KEEP_ALIVE_TIMEOUT_S", "0"), ("WEDGE_RESTART_S", "900"),
    ("SPAWN_READY_TIMEOUT_S", "360"),
    ("SHM_ORPHAN_MAX_AGE_S", "300"), ("METRICS_ENABLED", "1"),
    ("MAX_AUDIO_MB", "200"),
    ("IPC_REDIS_SOCKET", "/tmp/asr-redis.sock"),
    ("IPC_REDIS_URL", ""),
    # ── 功能开关与设备 ──
    ("DEVICE", ""), ("ASCEND_DEVICE_ID", ""),
    # ── 模型与模式 ──
    ("MODE", "mixed"),
    # ── 平台 ──
    ("PORT", "10095"), ("ADMIN_PORT", "10096"), ("LOG_LEVEL", "INFO"),
]


_ENV_NOISE = {"PATH", "HOME", "HOSTNAME", "PWD", "SHLVL", "_", "LANG", "TERM",
              "OLDPWD", "LC_ALL", "PYTHON_VERSION", "GPG_KEY", "SSL_CERT_DIR",
              "SSL_CERT_FILE", "container", "MOTD_SHOWN", "INVOCATION_ID",
              "JOURNAL_STREAM"}


def _mask_env(name: str, val: str) -> str:
    """密钥类变量值打码（按变量名关键词，不猜内容）。"""
    up = name.upper()
    if any(s in up for s in ("PASSWORD", "PASSWD", "TOKEN", "SECRET", "KEY")):
        return "***"
    return val


def dump_config() -> None:
    """启动自证：print 直写 stdout（不受 LOG_LEVEL 影响），supervisor 启动
    第一件事调用。三段——
    ① 可调配参数实际值（env 覆盖后 vs 默认值；redis URL 密码打码）
    ② 内部注入变量（勿配置）
    ③ 其余环境变量全量（抓拼错的变量名/未登记的配置；密钥打码）
    """
    out = ["──── tunable params (name=value [default]) ────"]
    for name, default in TUNABLE_PARAMS:
        val = os.getenv(name, default)
        if name == "IPC_REDIS_URL" and "://" in val and "@" in val:
            # redis://:password@host:port —— 密码不进日志
            scheme, rest = val.split("://", 1)
            val = f"{scheme}://***@{rest.rsplit('@', 1)[-1]}"
        mark = "" if val == default else "  <- overridden"
        out.append(f"  {name}={val} [{default}]{mark}")
    out.append("──── internal injection (do not configure) ────")
    for name in ("WORKER_TAG", "ASR_HEALTH_DIR", "IMAGE_VERSION"):
        out.append(f"  {name}={os.getenv(name, '<unset>')}")
    known = {n for n, _d in TUNABLE_PARAMS} | {
        "WORKER_TAG", "ASR_HEALTH_DIR", "IMAGE_VERSION", "ASR_URL"}
    out.append("──── other env（未登记变量排查用；密钥已打码）────")
    for name in sorted(k for k in os.environ
                       if k not in _ENV_NOISE and k not in known):
        out.append(f"  {name}={_mask_env(name, os.environ[name])}")
    out.append("──── end of config dump ───")
    print("\n".join(out), flush=True)


# env 解析告警走独立 logger（调用多发生在 setup_logging 之前/之外）
_cfg_log = logging.getLogger("asr.config")


def log_level(default: str = "INFO") -> int:
    """LOG_LEVEL 健壮解析：非法值回退默认，绝不在 import/启动期抛异常。"""
    raw = (os.getenv("LOG_LEVEL") or default).strip().upper()
    level = getattr(logging, raw, None)
    if not isinstance(level, int):
        _cfg_log.warning("LOG_LEVEL=%r 非法，使用默认 %s", raw, default)
        return getattr(logging, default.upper(), logging.INFO)
    return level


def log_level_name(default: str = "info") -> str:
    """uvicorn 用的级别名（它要字符串）；非法值回退默认。"""
    raw = (os.getenv("LOG_LEVEL") or default).strip().lower()
    if raw == "warn":
        return "warning"
    if raw in ("critical", "error", "warning", "info", "debug", "trace"):
        return raw
    _cfg_log.warning("LOG_LEVEL=%r 非法（uvicorn），使用默认 %s", raw, default)
    return default


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    """int(os.getenv(...)) 的健壮版：坏值告警 + 回退默认（生产可用性优先）。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        _cfg_log.warning("env %s=%r 非整数，使用默认值 %s", name, raw, default)
        return default
    if minimum is not None and val < minimum:
        _cfg_log.warning("env %s=%d 低于下限 %s，钳制为下限", name, val, minimum)
        return minimum
    return val


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    """同 env_int 的浮点版。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        _cfg_log.warning("env %s=%r 非数值，使用默认值 %s", name, raw, default)
        return default
    if minimum is not None and val < minimum:
        _cfg_log.warning("env %s=%s 低于下限 %s，钳制为下限", name, val, minimum)
        return minimum
    return val


class BusyWatch:
    """推理卡死看门狗的工人侧状态。

    任务从队列取出时 enter()、处理完毕 exit()；心跳快照读 stuck_s() 上报
    "busy 已持续多久"。非 busy 恒 0——闲置工人绝不会被误判；busy 期间
    心跳线程照常落盘（mtime 新鲜），但 stuck_s 持续增长，supervisor 据此
    识别"心跳活着、主循环 hang 死"（NPU aicore 挂起、funasr 推理卡死等
    mtime 检测盲区）。monotonic 计时不受系统时钟跳变影响。"""

    __slots__ = ("_since",)

    def __init__(self):
        self._since = None

    def enter(self) -> None:
        self._since = time.monotonic()

    def exit(self) -> None:
        self._since = None

    def stuck_s(self) -> float:
        since = self._since
        return round(time.monotonic() - since, 1) if since is not None else 0.0


_wtag_filter = None  # 单例可变 tag 过滤器（幂等关键：重复调用不堆积）


def setup_logging(tag: str) -> logging.Logger:
    """带 [角色标签] 前缀的日志初始化（幂等，重复调用安全）。"""
    global _wtag_filter

    class _WTagFilter(logging.Filter):
        def filter(self, record):
            record.wtag = self.tag
            return True

    logging.basicConfig(level=log_level(),
                        format="[%(wtag)s] %(asctime)s "
                               "%(levelname)s:%(name)s:%(message)s")
    if _wtag_filter is None:
        _wtag_filter = _WTagFilter()
        for h in logging.getLogger().handlers:
            h.addFilter(_wtag_filter)
    _wtag_filter.tag = tag
    return logging.getLogger("asr." + tag)


def write_heartbeat(health_dir: str, pid: int, snapshot: dict) -> None:
    """心跳落盘（原子替换）：worker-<pid>.json，supervisor 据此做 hang 检测。
    pid/ts 由本函数统一注入（调用方快照可省）。"""
    os.makedirs(health_dir, exist_ok=True)
    snapshot.setdefault("pid", int(pid))
    snapshot.setdefault("ts", time.time())
    tmp = os.path.join(health_dir, f".tmp-{pid}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(health_dir, f"worker-{pid}.json"))


def _heartbeat_once(tag: str, pid: int, stats_fn, health_dir: str) -> None:
    """单次心跳：快照组装 + 原子落盘。失败向上抛（由循环侧计频告警）。"""
    snap = {"wtag": tag, "pid": pid, "ts": time.time()}
    snap.update(stats_fn())
    write_heartbeat(health_dir, pid, snap)


def heartbeat_loop(tag: str, pid: int, stats_fn, health_dir: str = None,
                   interval: float = 5.0):
    """心跳线程主体：周期落盘 stats_fn() 快照。持续失败只告警不打死
    （第 1 次与每第 12 次打 warning，≈每分钟一条）。"""
    health_dir = health_dir or HEALTH_DIR
    os.makedirs(health_dir, exist_ok=True)
    fails = 0
    while True:
        try:
            _heartbeat_once(tag, pid, stats_fn, health_dir)
            fails = 0
        except Exception as exc:  # noqa: BLE001
            fails += 1
            if fails == 1 or fails % 12 == 0:
                logging.getLogger("asr.heartbeat").warning(
                    "[%s] heartbeat write failed (%d consecutive): %s",
                    tag, fails, exc)
        time.sleep(interval)


def start_heartbeat(tag: str, pid: int, stats_fn, health_dir: str = None,
                    interval: float = 5.0):
    import threading
    t = threading.Thread(target=heartbeat_loop,
                         args=(tag, pid, stats_fn, health_dir, interval),
                         daemon=True)
    t.start()
    return t


# ── 心跳/SHM 观测助手（gateway 聚合端点与 supervisor 看门狗共用）──


def heartbeat_files(health_dir: str = None, max_age_s: float = None) -> list:
    """健康目录中新鲜的 worker-*.json 文件路径（mtime 超 max_age_s 跳过，
    None=不过滤）。目录读不了返回空——观测失败绝不炸调用方。"""
    health_dir = health_dir or HEALTH_DIR
    try:
        names = os.listdir(health_dir)
    except OSError:
        return []
    out = []
    now = time.time()
    for name in names:
        if not (name.startswith("worker-") and name.endswith(".json")):
            continue
        path = os.path.join(health_dir, name)
        try:
            if max_age_s is not None and now - os.path.getmtime(path) > max_age_s:
                continue
        except OSError:
            continue
        out.append(path)
    return out


def iter_heartbeats(health_dir: str = None, max_age_s: float = None) -> list:
    """读取新鲜心跳文件为 dict 列表（损坏文件跳过）。"""
    out = []
    for path in heartbeat_files(health_dir, max_age_s):
        try:
            with open(path, encoding="utf-8") as f:
                out.append(json.load(f))
        except (OSError, ValueError):
            continue
    return out


def shm_usage() -> dict | None:
    """/dev/shm 使用情况（字节 + MB + 水位 pct）。不可用（非 POSIX/Windows
    开发机）返回 None。gateway 观测端点与 supervisor 启动自检共用。"""
    try:
        st = os.statvfs("/dev/shm")
        total = st.f_frsize * st.f_blocks
        if total <= 0:
            return None
        used = st.f_frsize * (st.f_blocks - st.f_bavail)
        return {"total": total, "used": used,
                "total_mb": round(total / 2**20, 1),
                "used_mb": round(used / 2**20, 1),
                "pct": round(used / total * 100, 1)}
    except (OSError, AttributeError):
        return None
