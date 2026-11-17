"""静态守卫：源码级扫描，防“pytest/QEMU 不执行入口函数 → 服务器启动才崩”类回归。

真实事故（20261118）两连：
- supervisor.main() 残留 proc_common.start_redis_server——pytest/import/QEMU
  全不执行 main()，66 用例全绿但容器启动即崩；
- tests 顶层 import 真 torch——QEMU 下 aarch64 C 扩展段错误。
本文件不依赖执行路径：AST 解析全部生产模块源码逐引用核对 + 清单断言。"""
import ast
import inspect

import compute
import gateway
import supervisor

import audio
import asr_common
import ipc
import proc_common
import protocol

# 传输层已迁 ipc.py 的符号（proc_common 上必须保持不存在）
_MOVED = [
    "start_redis_server", "AsrClaimQueue", "InstrumentedQueue",
    "_make_redis_client", "shared_redis", "mailbox_depths",
    "wait_stats", "push_wait_sample", "cancel_add", "is_cancelled",
    "task_score", "decode_member", "make_worker_id",
]

# 全部生产模块（测试模块不参与：它们允许 mock 注入）
_MODULES = {
    "audio": audio, "gateway": gateway, "compute": compute,
    "supervisor": supervisor, "ipc": ipc, "protocol": protocol,
    "proc_common": proc_common, "asr_common": asr_common,
}


def test_moved_symbols_absent_from_proc_common():
    for sym in _MOVED:
        assert not hasattr(proc_common, sym), \
            f"proc_common.{sym} 不应存在（已迁 ipc.py）"


def test_no_moved_symbol_referenced_via_proc_common():
    for mname, mod in _MODULES.items():
        src = inspect.getsource(mod)
        for sym in _MOVED:
            assert f"proc_common.{sym}" not in src, (
                f"{mname} 引用 proc_common.{sym}——传输层已迁 ipc.py，"
                f"请改为 ipc.{sym}（真实事故：main() 残留旧路径启动即崩）")


def test_ipc_actually_has_moved_symbols():
    for sym in _MOVED:
        assert hasattr(ipc, sym), f"ipc.{sym} 缺失——_MOVED 清单与实际漂移"


def test_all_cross_module_attr_references_exist():
    """AST 终极防线：扫描每个生产模块源码里所有 `<项目模块>.属性` 引用，
    按 (owner, attr) 成对核对目标模块真实拥有该属性——覆盖 main()/
    gateway_main/_claim_loop 等任何测试执行不到的函数体，AttributeError
    类启动崩溃在本地就地炸出。"""
    for mname, mod in _MODULES.items():
        tree = ast.parse(inspect.getsource(mod))
        refs = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in _MODULES):
                refs.add((node.value.id, node.attr))
        missing = sorted(f"{m}.{a}" for m, a in refs
                         if not hasattr(_MODULES[m], a))
        assert not missing, \
            f"模块 {mname} 引用了不存在的属性 {missing}——启动期 AttributeError 隐患"
