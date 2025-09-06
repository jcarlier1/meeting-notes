[CmdletBinding()]
param(
  [Parameter(Mandatory = $true, HelpMessage = "Path to audio/video file to transcribe")]
  [string]$Audio,

  [Parameter(Mandatory = $false, HelpMessage = "Output directory inside workspace")]
  [string]$Out = "data/tmp/run_nemo"
)

function Write-Info($msg) { Write-Host "[nemo] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[nemo] $msg" -ForegroundColor Yellow }

try { $null = & docker --version } catch { Write-Error "Docker is required but not found."; exit 1 }

$image = "nvcr.io/nvidia/nemo:25.07"

# Resolve paths
$workspace = (Resolve-Path -LiteralPath ".").Path
$audioAbs = (Resolve-Path -LiteralPath $Audio).Path
$audioDir = Split-Path -Parent $audioAbs
$audioFile = Split-Path -Leaf $audioAbs

# Ensure output directory exists in workspace
$outHost = Join-Path $workspace $Out
New-Item -ItemType Directory -Force -Path $outHost | Out-Null

Write-Info "Workspace: $workspace"
Write-Info "Audio: $audioAbs"
Write-Info "Output: $outHost"

Write-Warn "If you haven't already, run: docker login nvcr.io (username: \$oauthtoken, password: your NGC API key)"
Write-Info "Pulling image (if not present): $image"
try { & docker pull $image } catch { Write-Warn "Pull failed or skipped; continuing if already present." }

# Compose docker run args
$args = @(
  "run","--rm","-it",
  "--gpus","all",
  "--shm-size","8g",
  "-v", "$workspace:/workspace",
  "-v", "$audioDir:/data:ro",
  "-w", "/workspace",
  $image,
  "bash","-lc",
  # Install ffmpeg (for resampling/chunking) then run the CLI inside the container
  "apt-get update && apt-get install -y -qq ffmpeg >/dev/null && python -m meeting_notes.cli asr-file '/data/$audioFile' --out '/workspace/$Out'"
)

Write-Info "Starting NeMo container and running transcription..."
& docker @args

if ($LASTEXITCODE -ne 0) {
  Write-Error "NeMo transcription failed. Check the output above."
  exit $LASTEXITCODE
}

Write-Info "Done. Transcripts are in $outHost"

