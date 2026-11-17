"""audio.py 解码链测试：wav 多位深（含 24bit）、重采样、REST 两级链、ffmpeg 兜底层。"""
import io

import numpy as np
import pytest
import soundfile as sf

import audio


def _wav_bytes(sr=16000, seconds=0.5, subtype="PCM_16", freq=440.0):
    x = (0.3 * np.sin(2 * np.pi * freq * np.arange(int(sr * seconds)) / sr)
         ).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, subtype=subtype, format="WAV")
    return buf.getvalue()


def test_decode_audio_wav_soundfile_tier():
    wave = audio.decode_audio(_wav_bytes())
    assert wave.dtype == np.float32 and wave.size == 8000   # 0.5s @16k
    assert 0.0 < float(np.abs(wave).max()) <= 1.0


def test_decode_audio_resampled_to_16k():
    wave = audio.decode_audio(_wav_bytes(sr=8000))
    assert wave.size == 8000          # 0.5s 音频，重采样后 16k 采样点数不变


def test_parse_decode_wav_bitdepths():
    """16/8/24/32bit + float32 wav 全部位深走手写链。"""
    for subtype in ("PCM_16", "PCM_U8", "PCM_24", "PCM_32", "FLOAT"):
        raw = _wav_bytes(subtype=subtype)
        fmt, off = audio.parse_wav(raw)
        assert fmt is not None and fmt["fs"] == 16000 and fmt["bits"] in (8, 16, 24, 32)
        x = audio.decode_wav(raw[off:], fmt)
        assert x.dtype == np.int16 and x.size == 8000


def test_decode_audio_ffmpeg_tier():
    """mp3 只有 ffmpeg 兜底层能解（libsndfile 1.0.x 不认）；本机没 ffmpeg 则跳过。"""
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        pytest.skip("本机未装 ffmpeg（镜像内必装）")
    mp3 = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-ar", "16000", "-ac", "1", "-f", "mp3", "pipe:1"],
        capture_output=True, timeout=30).stdout
    assert mp3
    wave = audio.decode_audio(mp3)
    # mp3 编码器延迟 + 帧对齐（1152 样本/帧）会引入 ±样本，不能断言精确 16000
    assert wave.dtype == np.float32
    assert 14000 <= len(wave) <= 19000
    assert float(np.abs(wave).max()) > 0.05     # 解码出真实音频内容（非全静音）


def test_decode_audio_garbage_raises():
    with pytest.raises(Exception):
        audio.decode_audio(b"this is not audio at all")


def test_batch_bucket():
    from compute import _batch_bucket
    assert [_batch_bucket(n) for n in (1, 2, 5, 9, 17, 32)] == \
        ["1", "2-4", "5-8", "9-16", "17+", "17+"]
