"""静态守卫：传输层符号已迁 ipc.py——源码级扫描防“main() 残留旧路径”类回归。

真实事故（20261118）：supervisor.main() 残留 proc_common.start_redis_server，
pytest/import/QEMU 全不执行 main()，66 用例全绿但容器启动即崩。本测试不依赖
执行路径，直接扫源码——重构后任何旧路径引用当场暴露。
"""
import inspect

import compute
import gateway
import supervisor

# 传输层已迁 ipc.py 的符号（proc_common 上必须保持不存在）
_MOVED = [
    "start_redis_server", "AsrClaimQueue", "InstrumentedQueue",
    "_make_redis_client", "shared_redis", "mailbox_depths",
    "wait_stats", "push_wait_sample", "cancel_add", "is_cancelled",
    "task_score", "decode_member", "make_worker_id",
]


def test_moved_symbols_absent_from_proc_common():
    for sym in _MOVED:
        assert not hasattr(supervisor.proc_common, sym), \
            f"proc_common.{sym} 不应存在（已迁 ipc.py）"


def test_no_moved_symbol_referenced_via_proc_common():
    for mod in (gateway, compute, supervisor):
        src = inspect.getsource(mod)
        for sym in _MOVED:
            assert f"proc_common.{sym}" not in src, (
                f"{mod.__name__} 引用 proc_common.{sym}——传输层已迁 ipc.py，"
                f"请改为 ipc.{sym}（真实事故：main() 残留旧路径启动即崩）")


def test_ipc_actually_has_moved_symbols():
    import ipc
    for sym in _MOVED:
        assert hasattr(ipc, sym), f"ipc.{sym} 缺失——_MOVED 清单与实际漂移"
