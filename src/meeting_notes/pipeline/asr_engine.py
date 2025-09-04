from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .audio_utils import ffmpeg_resample_to_wav, record_chunks
from .io_utils import load_yaml
from .stitcher import merge_segments


ASR_CONFIG_PATH = Path("configs/asr.yaml")


def _load_asr_config() -> Dict:
    return load_yaml(ASR_CONFIG_PATH)


class _ASRBackend:
    def __init__(self, model_name: str, device: str, vad: bool, model_dir: Optional[str] = None, language: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.vad = vad
        self.language = language
        self.model_dir = model_dir
        self.backend = None
        self.backend_name = None
        self._init_backend()

    def _init_backend(self):
        # Try CrisperWhisper first
        try:
            import crisper_whisper as cw  # type: ignore

            # Hypothetical initialization API - if unavailable, falls back
            self.backend = cw.load_model(
                self.model_name,
                device=self.device if self.device != "auto" else None,
                model_dir=self.model_dir,
            )
            self.backend_name = "crisper-whisper"
            return
        except Exception:
            pass

        # Fallback to faster-whisper
        from faster_whisper import WhisperModel  # type: ignore

        kwargs = {}
        if self.model_dir:
            kwargs["download_root"] = self.model_dir
        dev = None if self.device == "auto" else self.device
        self.backend = WhisperModel(self.model_name, device=dev or "cpu", compute_type="int8", **kwargs)
        self.backend_name = "faster-whisper"

    def transcribe(self, wav_path: str, language: Optional[str] = None, vad: Optional[bool] = None) -> List[Dict]:
        language = language or self.language
        vad = self.vad if vad is None else vad

        if self.backend_name == "crisper-whisper":
            # Hypothetical API
            result = self.backend.transcribe(wav_path, language=language, vad=vad)
            segments = [
                {"start": float(s.start), "end": float(s.end), "text": s.text}
                for s in result.segments
            ]
            return segments

        # faster-whisper path
        gen, info = self.backend.transcribe(
            wav_path,
            language=None if (language in (None, "auto")) else language,
            vad_filter=bool(vad),
            word_timestamps=False,
        )
        segments = []
        for s in gen:
            segments.append({
                "start": float(s.start or 0.0),
                "end": float(s.end or 0.0),
                "text": s.text.strip(),
            })
        return segments


def transcribe_file(path: str | Path) -> List[Dict]:
    cfg = _load_asr_config()
    model = _ASRBackend(
        model_name=cfg.get("model", "large-v3"),
        device=cfg.get("device", "auto"),
        vad=bool(cfg.get("vad", True)),
        model_dir=str(Path("data/models/asr").absolute()),
        language=cfg.get("language", "auto"),
    )

    # Convert input to 16k mono WAV
    tmp_dir = Path("data/tmp/asr"); tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_dir / "input.wav"
    ffmpeg_resample_to_wav(path, wav_path)

    # If chunking requested, slice via ffmpeg and stitch results
    ch = cfg.get("chunking", {})
    seg_sec = int(ch.get("segment_sec", 20))
    ov_sec = int(ch.get("overlap_sec", 2))

    if seg_sec > 0:
        segments: List[Dict] = []
        duration = _probe_duration(wav_path)
        step = seg_sec - ov_sec
        t = 0.0
        idx = 0
        while t < duration:
            start = max(0.0, t - ov_sec)
            end = min(duration, t + seg_sec)
            chunk_path = tmp_dir / f"chunk_{idx:04d}.wav"
            _ffmpeg_trim(wav_path, chunk_path, start, end - start)
            new = model.transcribe(str(chunk_path), language=cfg.get("language"), vad=cfg.get("vad"))
            # Offset timestamps
            for s in new:
                s["start"] += start
                s["end"] += start
            segments = merge_segments(segments, new)
            idx += 1
            t += step
        return segments
    else:
        return model.transcribe(str(wav_path), language=cfg.get("language"), vad=cfg.get("vad"))


def transcribe_live(outdir: Path, duration_sec: Optional[int] = None) -> List[Dict]:
    cfg = _load_asr_config()
    model = _ASRBackend(
        model_name=cfg.get("model", "large-v3"),
        device=cfg.get("device", "auto"),
        vad=bool(cfg.get("vad", True)),
        model_dir=str(Path("data/models/asr").absolute()),
        language=cfg.get("language", "auto"),
    )
    ch = cfg.get("chunking", {})
    seg_sec = int(ch.get("segment_sec", 20))
    ov_sec = int(ch.get("overlap_sec", 2))

    chunks_dir = outdir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_segments: List[Dict] = []
    try:
        for chunk in record_chunks(chunks_dir, seg_sec, ov_sec, max_duration_sec=duration_sec, sr=16000):
            new = model.transcribe(str(chunk), language=cfg.get("language"), vad=cfg.get("vad"))
            # No offset needed; chunk files already include overlap context
            all_segments = merge_segments(all_segments, new)
            # Write rolling outputs
            (outdir / "transcript.json").write_text(
                __to_json({"segments": all_segments})
            )
            (outdir / "transcript.txt").write_text(
                "\n".join(f"[{_fmt_ts(s['start'])}-{_fmt_ts(s['end'])}] {s['text']}" for s in all_segments)
            )
            print(f"[live] segments: {len(all_segments)} | last chunk: {chunk.name}")
    except KeyboardInterrupt:
        print("[live] Stopped by user")
    return all_segments


def _probe_duration(wav_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def _ffmpeg_trim(src: Path, dst: Path, start: float, duration: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ss",
        f"{start}",
        "-t",
        f"{duration}",
        "-c",
        "copy",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def __to_json(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


def _fmt_ts(x: float) -> str:
    m, s = divmod(int(x), 60)
    return f"{m:02d}:{s:02d}"

