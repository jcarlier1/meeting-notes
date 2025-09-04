from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .io_utils import load_yaml


from pathlib import Path
from importlib.resources import files


LLM_CONFIG_PATH = Path("configs/llm.yaml")


def summarize(segments: List[Dict]) -> Tuple[Dict, str]:
    cfg = load_yaml(LLM_CONFIG_PATH)
    # Load prompts from package resources
    system_prompt = files("meeting_notes.prompts").joinpath("meeting_notes_system.txt").read_text().strip()  # type: ignore
    user_template = files("meeting_notes.prompts").joinpath("meeting_notes_user.txt").read_text()  # type: ignore

    transcript_text = _segments_to_text(segments)
    user_prompt = user_template.replace("{{TRANSCRIPT}}", transcript_text)

    backend = cfg.get("backend", "llama-cpp")
    if backend != "llama-cpp":
        raise ValueError("Only llama-cpp backend is supported in this project")

    model_path = Path("data/models/llm") / cfg.get("model")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run scripts/download_models.sh")

    from llama_cpp import Llama

    ctx = int(cfg.get("context", 4096))
    n_gpu_layers = int(cfg.get("gpu_layers", 0))
    samp = cfg.get("sampling", {})
    temperature = float(samp.get("temperature", 0.2))
    top_p = float(samp.get("top_p", 0.9))
    max_tokens = int(samp.get("max_tokens", 1024))

    llm = Llama(
        model_path=str(model_path),
        n_ctx=ctx,
        n_gpu_layers=n_gpu_layers,
        chat_format="qwen2",
        verbose=False,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    out = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    text = out["choices"][0]["message"]["content"].strip()

    notes = _parse_notes(text)
    notes["model_info"] = {
        "backend": backend,
        "model": str(model_path.name),
        "context": ctx,
        "gpu_layers": n_gpu_layers,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    }
    notes["markdown"] = text

    return notes, text


def _segments_to_text(segments: List[Dict]) -> str:
    lines = []
    for s in segments:
        start = _fmt_ts(s.get("start", 0.0))
        text = s.get("text", "").strip()
        lines.append(f"[{start}] {text}")
    return "\n".join(lines)


def _fmt_ts(x: float) -> str:
    m, s = divmod(int(x), 60)
    return f"{m:02d}:{s:02d}"


def _parse_notes(text: str) -> Dict:
    # Heuristic parser for the three sections
    sections = {
        "main_points": [],
        "discussion_points": [],
        "action_items": [],
    }

    current = None
    for line in text.splitlines():
        low = line.lower().strip()
        if low.startswith("main points"):
            current = "main_points"; continue
        if low.startswith("discussion points"):
            current = "discussion_points"; continue
        if low.startswith("action items"):
            current = "action_items"; continue
        if line.strip().startswith(('-', '*')) and current:
            item = line.strip()[1:].strip()
            sections[current].append(item)

    return sections
