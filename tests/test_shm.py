"""SHM 块生命周期测试（真 multiprocessing.shared_memory，Windows 可跑）：
写→读往返（int16 线格式）、幂等清理、worker 视角读取。"""
import numpy as np
import pytest

import gateway


@pytest.fixture()
def clean_blocks():
    created = []
    yield created
    for b in created:
        gateway._unlink_block(dict(b))


def test_write_read_roundtrip(clean_blocks):
    clips = [np.full(1600, 0.5, dtype=np.float32),
             np.full(3200, -0.25, dtype=np.float32)]
    block = gateway._write_shm_block(clips)
    clean_blocks.append(dict(block))
    assert block["segs"][0] == {"idx": 0, "offset": 0, "samples": 1600}
    assert block["segs"][1] == {"idx": 1, "offset": 3200, "samples": 3200}
    # worker 视角：按 piece 描述符读取（int16 → float32 /32767）
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=block["shm_name"])
    try:
        for seg, clip in zip(block["segs"], clips):
            piece = dict(seg, shm_name=block["shm_name"])
            x = np.frombuffer(
                shm.buf[piece["offset"]:piece["offset"] + piece["samples"] * 2],
                dtype="<i2").astype(np.float32) / 32767.0
            assert len(x) == len(clip)
            assert np.allclose(x, clip, atol=2 / 32767.0)   # 16bit 量化误差内
    finally:
        shm.close()


def test_unlink_idempotent(clean_blocks):
    block = gateway._write_shm_block([np.zeros(100, dtype=np.float32)])
    clean_blocks.append(dict(block))
    gateway._unlink_block(block)       # 第一次：清掉
    gateway._unlink_block(block)       # 第二次：幂等不抛
    gateway._unlink_block(None)        # None 安全
    from multiprocessing import shared_memory
    with pytest.raises((FileNotFoundError, OSError)):
        shared_memory.SharedMemory(name=clean_blocks[-1]["shm_name"])


def test_empty_pieces_safe(clean_blocks):
    block = gateway._write_shm_block([])
    clean_blocks.append(dict(block))
    assert block["segs"] == []
