# Meeting Notes CLI

CLI-only meeting notetaker that performs live and offline transcription with NVIDIA NeMo (containerized) and summarizes with a local LLM (Qwen2.5-3B-Instruct via llama.cpp) into Main points, Discussion points, and Action items.

## Features
- Offline file transcription with VAD and chunking
- Live microphone capture with rolling transcripts
- Local summarization using `llama.cpp` bindings
- Fully containerized (CPU and NVIDIA GPU Dockerfiles)

## NeMo ASR (Container)

This repository is configured to use the NVIDIA NeMo pre-built container for ASR.

Prerequisites:
- NVIDIA GPU + drivers, Docker Desktop, and NVIDIA Container Toolkit
- NGC account + API key (https://ngc.nvidia.com)

One-time setup:
```powershell
docker login nvcr.io   # username: $oauthtoken, password: <your NGC API key>
docker pull nvcr.io/nvidia/nemo:25.07
```

Run transcription from PowerShell (Windows):
```powershell
scripts/run_nemo_asr.ps1 -Audio 
```

Outputs go to `data/tmp/run_nemo` by default. The script mounts your workspace and audio folder into the container, installs `ffmpeg` inside the container for resampling/chunking, and runs the CLI `asr-file` command with the NeMo backend.

ASR configuration (`configs/asr.yaml`) now includes:
```yaml
backend: nemo
model: stt_en_fastconformer_transducer_large
```

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

This downloads (for the legacy Whisper path):
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

## Notes on Backends

- Default ASR backend is now NeMo when running via the NeMo container (`configs/asr.yaml: backend: nemo`).
- If you run the old Dockerfiles (CPU/GPU) without NeMo, the pipeline will automatically fall back to CrisperWhisper (if installed) or Faster-Whisper.

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
