"""supervisor.main() 全链冒烟（mock 重依赖）：防启动期 AttributeError 回归。

真实事故（20261118）：main() 里残留 proc_common.start_redis_server()（传输层
已迁 ipc.py），pytest/import/QEMU 容器全不执行 main()，66 用例全绿但容器
启动即崩。本测试真调 main() 全链到主循环第一步——若任何属性引用缺失当场
AttributeError，且 redis 拉起必须走 ipc（旧路径 proc_common 上已无此函数，
残留即炸）。
"""
import pytest

import supervisor


def test_supervisor_main_smoke(monkeypatch, capsys):
    # 重依赖 mock：redis 拉起（注意 patch 在 ipc 上——回归点）/ 心跳 / 容量 / 编排
    fake_proc = type("FakeProc", (), {"pid": 424242})()
    monkeypatch.setattr(supervisor.ipc, "start_redis_server",
                        lambda: fake_proc)
    monkeypatch.setattr(supervisor.proc_common, "start_heartbeat",
                        lambda **kw: None)
    monkeypatch.setattr(supervisor, "_capacity_check", lambda: None)
    monkeypatch.setattr(supervisor, "spawn_all", lambda: None)

    def _boom(_s):
        raise KeyboardInterrupt          # 主循环第一步即中断：全链已走完
    monkeypatch.setattr(supervisor.time, "sleep", _boom)

    with pytest.raises(KeyboardInterrupt):
        supervisor.main()

    out = capsys.readouterr().out
    assert "tunable params" in out                       # 启动自证先于一切
    assert supervisor._task_queues["asr"]._key == "asr:q:1:asr"   # mailbox 已建
    assert supervisor._task_queues["vad"]._key == "asr:q:0:vad"
