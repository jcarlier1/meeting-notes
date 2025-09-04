from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Generator, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None  # allow environments without audio


def ffmpeg_resample_to_wav(src: str | Path, dst_wav: str | Path, sr: int = 16000) -> None:
    dst = Path(dst_wav)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sr),
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mic_stream(sr: int = 16000, block_sec: float = 1.0) -> Generator[np.ndarray, None, None]:
    if sd is None:
        raise RuntimeError("sounddevice is not available in this environment")
    blocksize = int(sr * block_sec)
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=blocksize) as stream:
        while True:
            frames, _ = stream.read(blocksize)
            yield frames.copy()


def record_chunks(
    outdir: Path,
    segment_sec: int,
    overlap_sec: int,
    max_duration_sec: Optional[int] = None,
    sr: int = 16000,
) -> Generator[Path, None, None]:
    """Record microphone audio and write rolling WAV chunks to disk.

    Yields chunk file paths as they are finalized.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    if sd is None:
        raise RuntimeError("sounddevice is not available")

    ring = np.zeros(((segment_sec + overlap_sec) * sr, 1), dtype=np.float32)
    write_len = segment_sec * sr
    overlap_len = overlap_sec * sr
    started = time.time()
    idx = 0
    buf = []

    for block in mic_stream(sr=sr, block_sec=0.5):
        buf.append(block)
        if max_duration_sec and (time.time() - started) >= max_duration_sec:
            # Flush remaining as final chunk
            if buf:
                chunk = np.concatenate(buf, axis=0)
                ring = _append_ring(ring, chunk, sr)
                fpath = outdir / f"chunk_{idx:04d}.wav"
                _write_wav_numpy(ring[-(write_len + overlap_len) :], fpath, sr)
                yield fpath
            break

        # Emit a chunk every second if enough audio collected for segment
        total = sum(b.shape[0] for b in buf)
        if total >= sr:
            chunk = np.concatenate(buf, axis=0)
            buf.clear()
            ring = _append_ring(ring, chunk, sr)
            # When we accumulate at least (segment_sec) new seconds since last emit, write chunk
            emitted = outdir.glob("chunk_*.wav")
            count = len(list(emitted))
            # Heuristic: write every second once ring has enough samples
            if ring.shape[0] >= (write_len + overlap_len):
                fpath = outdir / f"chunk_{idx:04d}.wav"
                _write_wav_numpy(ring[-(write_len + overlap_len) :], fpath, sr)
                idx += 1
                yield fpath


def _append_ring(ring: np.ndarray, data: np.ndarray, sr: int) -> np.ndarray:
    new_len = ring.shape[0] + data.shape[0]
    if new_len <= ring.shape[0]:
        return np.concatenate([ring, data], axis=0)
    # keep last ring length
    concat = np.concatenate([ring, data], axis=0)
    return concat[-ring.shape[0] :]


def _write_wav_numpy(x: np.ndarray, path: Path, sr: int) -> None:
    # Write via ffmpeg to avoid extra deps
    tmp_raw = path.with_suffix(".raw")
    x.astype(np.float32).tofile(tmp_raw)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-i",
        str(tmp_raw),
        str(path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        tmp_raw.unlink()
    except Exception:
        pass

