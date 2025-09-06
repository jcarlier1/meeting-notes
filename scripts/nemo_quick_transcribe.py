#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Quick NeMo ASR transcription")
    p.add_argument("audio", help="Path to audio file")
    p.add_argument("--model", default="stt_en_fastconformer_transducer_large", help="NeMo model name")
    p.add_argument("--out", default=None, help="Optional output text file")
    args = p.parse_args()

    try:
        import nemo.collections.asr as nemo_asr  # type: ignore
    except Exception:
        raise SystemExit("NeMo not found. Run this inside the NeMo container: nvcr.io/nvidia/nemo:25.07")

    asr_model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    outs = asr_model.transcribe([args.audio], return_hypotheses=True)

    def _to_text(x) -> str:
        try:
            if isinstance(x, str):
                return x.strip()
            if isinstance(x, (list, tuple)) and x:
                return _to_text(x[0])
            txt = getattr(x, "text", None)
            if isinstance(txt, str):
                return txt.strip()
        except Exception:
            pass
        return str(x).strip()

    text = _to_text(outs[0]) if outs else ""
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"Wrote transcript to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

