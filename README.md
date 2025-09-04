# Meeting Notes CLI

CLI-only meeting notetaker that performs live and offline transcription with CrisperWhisper (fallback to Faster-Whisper) and summarizes with a local LLM (Qwen2.5-3B-Instruct via llama.cpp) into Main points, Discussion points, and Action items.

## Features
- Offline file transcription with VAD and chunking
- Live microphone capture with rolling transcripts
- Local summarization using `llama.cpp` bindings
- Fully containerized (CPU and NVIDIA GPU Dockerfiles)

## Quickstart

1) Build the Docker image (CPU):

```bash
docker build -f docker/Dockerfile.cpu -t meeting-notes:cpu .
```

GPU (requires NVIDIA Docker runtime and `--gpus all`):

```bash
docker build -f docker/Dockerfile.gpu -t meeting-notes:gpu .
```

2) Download models (one-time, networked):

```bash
docker run --rm -it -v $(pwd):/app -w /app meeting-notes:cpu bash scripts/download_models.sh
```

This downloads:
- ASR model `large-v3` (for Faster-Whisper)
- Qwen2.5-3B-Instruct GGUF quantized: `Qwen2.5-3B-Instruct.Q4_K_M.gguf`

3) Run transcription and summarization

File transcription:
```bash
docker run --rm -it -v $(pwd):/app -w /app meeting-notes:cpu \
  meeting-notes asr-file data/samples/sample.mp3 --out data/tmp/run1
```

Live transcription (press Ctrl+C to stop):
```bash
# You may need to pass audio devices from host to container
docker run --rm -it --device /dev/snd -v $(pwd):/app -w /app meeting-notes:cpu \
  meeting-notes asr-live --out data/tmp/live1
```

Summarize:
```bash
docker run --rm -it -v $(pwd):/app -w /app meeting-notes:cpu \
  meeting-notes summarize --transcript data/tmp/run1/transcript.json --out data/tmp/notes.md
```

Outputs are written to `data/tmp/...` as `.json`, `.txt`, and `.md`.

## Configuration

- ASR config: `configs/asr.yaml`
- LLM config: `configs/llm.yaml`

You can tweak model names, device, chunking, context length, and sampling params.

## Notes on CrisperWhisper

This project prefers CrisperWhisper if available. If the `crisper_whisper` package is not present, it falls back to `faster-whisper` with VAD filtering enabled. Ensure you run `scripts/download_models.sh` to prefetch needed model files so runtime remains offline.

## Development

Install editable (inside container):
```bash
bash scripts/bootstrap.sh
```

Run the CLI:
```bash
meeting-notes --help
```

## License

MIT

