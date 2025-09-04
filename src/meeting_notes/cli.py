import json
from pathlib import Path
from typing import Optional

import typer

from meeting_notes.pipeline.asr_engine import transcribe_file, transcribe_live
from meeting_notes.pipeline.summarizer import summarize


app = typer.Typer(help="Meeting notes CLI: ASR + summarization")


@app.command("asr-file")
def asr_file(
    path: str = typer.Argument(..., help="Path to media file (audio/video)"),
    out: str = typer.Option("data/tmp/run1", "--out", help="Output directory"),
):
    segments = transcribe_file(path)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "transcript.json").write_text(
        json.dumps({"segments": segments}, indent=2, ensure_ascii=False)
    )
    (outdir / "transcript.txt").write_text("\n".join(
        f"[{_fmt_ts(s['start'])}-{_fmt_ts(s['end'])}] {s['text']}" for s in segments
    ))
    print(f"Wrote transcripts to {outdir}")


@app.command("asr-live")
def asr_live(
    out: str = typer.Option("data/tmp/live1", "--out", help="Output directory"),
    duration: Optional[int] = typer.Option(None, help="Optional max duration in seconds"),
):
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    segments = transcribe_live(outdir, duration_sec=duration)
    (outdir / "transcript.json").write_text(
        json.dumps({"segments": segments}, indent=2, ensure_ascii=False)
    )
    (outdir / "transcript.txt").write_text("\n".join(
        f"[{_fmt_ts(s['start'])}-{_fmt_ts(s['end'])}] {s['text']}" for s in segments
    ))
    print(f"Wrote live transcripts to {outdir}")


@app.command("summarize")
def summarize_cmd(
    transcript: str = typer.Option(..., "--transcript", help="Path to transcript.json"),
    out: str = typer.Option("data/tmp/notes.md", "--out", help="Output markdown file"),
    out_json: str = typer.Option(
        "data/tmp/notes.json", "--out-json", help="Output notes JSON file"
    ),
):
    payload = json.loads(Path(transcript).read_text())
    notes_dict, notes_md = summarize(payload["segments"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(notes_md)
    Path(out_json).write_text(json.dumps(notes_dict, indent=2, ensure_ascii=False))
    print(f"Wrote notes to {out} and {out_json}")


def _fmt_ts(x: float) -> str:
    m, s = divmod(int(x), 60)
    return f"{m:02d}:{s:02d}"


def main():
    app()


if __name__ == "__main__":
    main()

