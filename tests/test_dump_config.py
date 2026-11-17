"""dump_config 启动自证测试：参数表/覆盖标记/密钥打码/全量环境段。"""
import proc_common


def test_dump_config_prints_tunables_and_overrides(capsys, monkeypatch):
    monkeypatch.setenv("ASR_WORKERS", "3")
    proc_common.dump_config()
    out = capsys.readouterr().out
    assert "ASR_WORKERS=3 [8]  <- overridden" in out   # env 覆盖 + 默认值对照
    assert "MODE=mixed [mixed]" in out                 # 未覆盖项
    assert "──── end of config dump ───" in out


def test_dump_config_masks_redis_password(capsys, monkeypatch):
    monkeypatch.setenv("IPC_REDIS_URL", "redis://:p@ssw0rd@host:6379")
    proc_common.dump_config()
    out = capsys.readouterr().out
    assert "redis://***@host:6379" in out
    assert "p@ssw0rd" not in out


def test_dump_config_masks_secret_env_and_lists_others(capsys, monkeypatch):
    monkeypatch.setenv("MY_DB_PASSWORD", "hunter2")
    monkeypatch.setenv("MY_CUSTOM_FLAG", "1")
    proc_common.dump_config()
    out = capsys.readouterr().out
    assert "MY_DB_PASSWORD=***" in out and "hunter2" not in out
    assert "MY_CUSTOM_FLAG=1" in out                   # 未登记变量也可见（抓拼写错误）


def test_dump_config_masks_env_with_log_level_warning(capsys, monkeypatch):
    """LOG_LEVEL=WARNING 时 dump 仍然可见（print 直写，不走日志级别）。"""
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    proc_common.dump_config()
    out = capsys.readouterr().out
    assert "tunable params" in out
