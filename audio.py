"""音频解码：wav 解析 + 多编码转换 + 16k 重采样（纯函数库，gateway 复用）。

两条解码路径（同一 16k/mono 目标口径）：
- REST 全量解码（decode_audio）：两级链，全部按字节内容嗅探（扩展名/媒体
  类型不参与）——① soundfile（libsndfile：wav/flac/ogg-vorbis…）→ scipy
  重采样；② ffmpeg 子进程兜底（mp3/m4a/aac/opus 全格式，直出 16k mono，
  镜像 apt 内置）。不经过 funasr 自带 loader：它顶层 import torchaudio
  （镜像刻意不装；1.4.21 容错但层①每次白抛异常），且行为随版本漂移。
- WS 流式（parse_wav/decode_wav/to16k）：手写 wav 头解析（支持头跨帧缓冲）+
  多位深 PCM 转换 + scipy 多相重采样（块边界轻微伪影 ASR 可容忍）。
不依赖 torchaudio（镜像内 torch 2.14 无配套轮子时的可用性保全）。
"""
import numpy as np

SAMPLE_RATE = 16000


def parse_wav(buf: bytes):
    """解析 wav 头。返回 (fmt, data偏移)；数据不足返回 (None, -1)（继续缓冲）；
    非 wav 抛 ValueError。fmt: dict(enc, ch, fs, bits)"""
    if len(buf) < 12:
        return None, -1
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise ValueError("not wav")
    off, fmt = 12, None
    while True:
        if off + 8 > len(buf):
            return None, -1
        cid = buf[off:off + 4]
        size = int.from_bytes(buf[off + 4:off + 8], "little")
        body = off + 8
        if cid == b"fmt ":
            if body + 16 > len(buf):
                return None, -1
            fmt = dict(enc=int.from_bytes(buf[body:body + 2], "little"),
                       ch=int.from_bytes(buf[body + 2:body + 4], "little"),
                       fs=int.from_bytes(buf[body + 4:body + 8], "little"),
                       bits=int.from_bytes(buf[body + 14:body + 16], "little"))
        elif cid == b"data":
            return (fmt, body) if fmt else (None, -1)
        off = body + size + (size & 1)      # chunk 2 字节对齐


def decode_wav(data: bytes, fmt: dict) -> np.ndarray:
    """wav 数据块 → int16 mono ndarray（按已解析的 fmt）"""
    enc, bits, ch = fmt["enc"], fmt["bits"], fmt["ch"]
    if enc == 1 and bits == 16:
        x = np.frombuffer(data, dtype=np.int16)
    elif enc == 1 and bits == 8:            # 8 位无符号
        x = (np.frombuffer(data, dtype=np.uint8).astype(np.int16) - 128) << 8
    elif enc == 1 and bits == 24:           # 24 位打包：取高 16 位
        b = np.frombuffer(data, dtype=np.uint8)
        if len(b) % 3:
            b = b[:len(b) // 3 * 3]
        u = (b[0::3].astype(np.uint32)
             | (b[1::3].astype(np.uint32) << 8)
             | (b[2::3].astype(np.uint32) << 16))
        x = ((u ^ 0x800000).astype(np.int32) - 0x800000 >> 8).astype(np.int16)
    elif enc == 1 and bits == 32:
        x = (np.frombuffer(data, dtype=np.int32) >> 16).astype(np.int16)
    elif enc == 3 and bits == 32:           # float32
        x = (np.frombuffer(data, dtype=np.float32) * 32767).astype(np.int16)
    else:
        raise ValueError(f"不支持的 wav 格式: enc={enc} bits={bits}（请转 16bit 或用 REST）")
    if ch > 1:                              # 多声道均值混音
        n = len(x) // ch
        if n:
            x = x[:n * ch].reshape(n, ch).mean(axis=1).astype(np.int16)
    return x


def to16k(pcm: np.ndarray, fs: int) -> np.ndarray:
    """int16 ndarray 重采样到 16k（scipy 多相滤波，块边界轻微伪影 ASR 可容忍）。"""
    if fs == SAMPLE_RATE or pcm.size == 0:
        return pcm
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(fs), SAMPLE_RATE)
    up, down = SAMPLE_RATE // g, int(fs) // g
    r = resample_poly(pcm.astype(np.float32) / 32768.0, up, down)
    return (r * 32768.0).astype(np.int16)


def decode_audio(data: bytes) -> np.ndarray:
    """上传音频字节 → 16k float32 波形（REST 用）。

    两级解码链（全按字节内容嗅探），两层都失败抛错（400/500 由调用方映射）：
    ① soundfile（libsndfile：wav/flac/ogg-vorbis…）
    ② ffmpeg 子进程（mp3/m4a/aac/opus 全格式；镜像 apt 内置，本地没装则
       mp3 类不可解——与镜像行为差异仅此一处）
    """
    try:
        import io
        import soundfile as sf
        x, fs = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return to16k((x.mean(axis=1) * 32768.0).astype(np.int16), fs).astype(
            np.float32) / 32768.0
    except Exception:   # noqa: BLE001  libsndfile 不认的容器（mp3/m4a/aac…）→ ffmpeg 兜底
        wave = _decode_audio_ffmpeg(data)
        if wave is None:
            raise
        return wave


def _decode_audio_ffmpeg(data: bytes) -> np.ndarray | None:
    """ffmpeg stdin → 任意容器解码为 16k mono float32。
    ffmpeg 缺失/超时/解码失败返回 None（调用方重抛原 soundfile 异常）。"""
    try:
        import subprocess
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-f", "f32le", "-ac", "1",
             "-ar", str(SAMPLE_RATE), "pipe:1"],
            input=data, capture_output=True, timeout=120)
        if out.returncode != 0 or not out.stdout:
            return None
        return np.frombuffer(out.stdout, dtype=np.float32).copy()
    except Exception:   # noqa: BLE001  FileNotFoundError（未装）/ 超时等
        return None
