from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# Lightweight shim that provides a minimal CrisperWhisper-like API
# by delegating to faster-whisper under the hood.


@dataclass
class _Segment:
    start: float
    end: float
    text: str


class _Result:
    def __init__(self, segments: List[_Segment]):
        self.segments = segments


class _CWModel:
    def __init__(self, model_name: str, device: Optional[str] = None, model_dir: Optional[str] = None):
        # Import lazily to keep module light
        from faster_whisper import WhisperModel  # type: ignore

        kwargs = {}
        if model_dir:
            kwargs["download_root"] = model_dir
        dev = device or "cpu"
        self._backend = WhisperModel(model_name, device=dev, compute_type="int8", **kwargs)

    def transcribe(self, audio_path: str, language: Optional[str] = None, vad: bool = True) -> _Result:
        # Map parameters to faster-whisper flags
        gen, _info = self._backend.transcribe(
            audio_path,
            language=None if (language in (None, "auto")) else language,
            vad_filter=bool(vad),
            word_timestamps=False,
        )
        segments: List[_Segment] = []
        for s in gen:
            segments.append(_Segment(start=float(s.start or 0.0), end=float(s.end or 0.0), text=s.text or ""))
        return _Result(segments)


def load_model(model_name: str, device: Optional[str] = None, model_dir: Optional[str] = None) -> _CWModel:
    """Return a model object with `.transcribe(...)`.

    This mirrors the hypothetical CrisperWhisper API expected by the app.
    """
    dev = None if device == "auto" else device
    return _CWModel(model_name, device=dev, model_dir=model_dir)

