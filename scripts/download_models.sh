#!/usr/bin/env bash
set -euo pipefail

ASR_DIR="data/models/asr"
LLM_DIR="data/models/llm"
mkdir -p "$ASR_DIR" "$LLM_DIR"

echo "[models] Prefetching Faster-Whisper 'large-v3' into $ASR_DIR"
python3 - <<'PY'
import os
os.makedirs("data/models/asr", exist_ok=True)
try:
    from faster_whisper import WhisperModel
    # This will download the model to download_root if not present
    _ = WhisperModel("large-v3", device="cpu", compute_type="int8", download_root="data/models/asr")
    print("Downloaded Faster-Whisper model: large-v3")
except Exception as e:
    print("Failed to pre-download Faster-Whisper model:", e)
PY

echo "[models] Downloading Qwen2.5-3B-Instruct GGUF (Q4_K_M) into $LLM_DIR"
QWEN_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf?download=true"
curl -L "$QWEN_URL" -o "$LLM_DIR/Qwen2.5-3B-Instruct.Q4_K_M.gguf"
echo "[models] Done."

