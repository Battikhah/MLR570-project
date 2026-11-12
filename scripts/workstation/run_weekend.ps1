<#
Run the PCD suite on the AUS GPU workstation (ESB-1043-20) for one access window.

Rules this script follows:
  * Access only Thursday 18:00 - Monday 08:00 (UAE time). The Python runner refuses to start
    outside the window and pauses with a checkpoint 20 minutes before it closes.
  * DeepFreeze wipes the local disk on any restart. Everything that matters (results, teacher
    weights, run checkpoints, logs, the CIFAR archives) is written to OneDrive; the local copy of
    the code, Python environment and datasets is rebuilt each session.

What it runs: the whole suite (all 18 families of the chosen profile) as one priority queue,
shared by -Procs processes on the GPU. Seed 0 of every family comes before any second seed, and
families run most-revealing first (src/pcdkd/experiments/suite.py, PRIORITY), so whatever finishes
by Monday is the most informative subset. Each teacher is trained just before the first run that
needs it. Rerunning the same command next window continues where it stopped.

Usage, in PowerShell on the workstation (connected over GlobalProtect + RDP):
  powershell -ExecutionPolicy Bypass -File run_weekend.ps1 -SetupOnly        # test access only
  powershell -ExecutionPolicy Bypass -File run_weekend.ps1                   # whole suite, 3 procs
  powershell -ExecutionPolicy Bypass -File run_weekend.ps1 -Procs 4 -Family "label-noise loss-scale"

When leaving, DISCONNECT the RDP session (close the window); do not sign out, which kills the run.
#>
param(
  [string]$Profile = "screen",
  [int]$Procs = 3,              # processes sharing the GPU (ResNet-8 students barely load one GPU)
  [string]$Family = "",         # optional subset, space-separated; default: every family
  [int]$Workers = 3,            # data-loading workers per process
  [string]$Sync = "",
  [switch]$SetupOnly
)
$ErrorActionPreference = "Stop"

# --- where things live ---------------------------------------------------------------------------
if (-not $Sync) {
  $od = if ($env:OneDriveCommercial) { $env:OneDriveCommercial } else { $env:OneDrive }
  if (-not $od) { throw "OneDrive is not signed in. Sign in with your AUS account, or pass -Sync <synced folder>." }
  $Sync = Join-Path $od "pcd-suite"
}
$Zip   = Join-Path $Sync "pcd-code.zip"
$Out   = Join-Path $Sync "outputs"      # results, teachers, checkpoints: survive DeepFreeze
$Logs  = Join-Path $Sync "logs"
$Cache = Join-Path $Sync "data-cache"   # CIFAR archives, so a wipe does not mean a re-download
$Work  = Join-Path $env:LOCALAPPDATA "pcd-suite"   # wiped by DeepFreeze; rebuilt below
$Code  = Join-Path $Work "code"
$Data  = Join-Path $Work "data"
if (-not (Test-Path $Zip)) { throw "Missing $Zip. On the Mac run scripts/workstation/pack.sh and upload the zip there." }
New-Item -ItemType Directory -Force $Out, $Logs, $Cache, $Work, $Data | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
Start-Transcript -Path (Join-Path $Logs "session-$stamp.log") | Out-Null

# --- 1. uv, per user (no admin rights needed) ----------------------------------------------------
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# --- 2. fresh code ---------------------------------------------------------------------------------
if (Test-Path $Code) { Remove-Item -Recurse -Force $Code }
Expand-Archive -Path $Zip -DestinationPath $Code
Set-Location $Code

# --- 3. environment: the locked packages, then the CUDA build of the same torch version ------------
# PyPI's Windows torch wheels are CPU-only. Every later command uses `uv run --no-sync` so uv does
# not put the CPU build back.
uv sync --frozen
$ver = uv run --no-sync python -c "import torch, torchvision as tv; print(torch.__version__.split('+')[0], tv.__version__.split('+')[0])"
$tv = $ver.Split(" ")
uv pip install --reinstall "torch==$($tv[0])" "torchvision==$($tv[1])" --index-url https://download.pytorch.org/whl/cu126
uv run --no-sync python -c "import sys, torch; ok = torch.cuda.is_available(); print('CUDA:', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA is not available to torch; check the GPU driver before running anything." }

# --- 4. sanity tests (CPU, synthetic data) ----------------------------------------------------------
uv run --no-sync pytest -q -m "not slow and not requires_data"
if ($LASTEXITCODE -ne 0) { throw "tests failed; not starting GPU runs" }

# --- 5. CIFAR-10/100: from the OneDrive cache, downloading once if it is empty ----------------------
foreach ($name in "cifar-10-python.tar.gz", "cifar-100-python.tar.gz") {
  $cached = Join-Path $Cache $name
  if (Test-Path $cached) { Copy-Item $cached (Join-Path $Data $name) }
}
uv run --no-sync python -c "from torchvision.datasets import CIFAR10, CIFAR100; import sys; [c(sys.argv[1], train=t, download=True) for c in (CIFAR10, CIFAR100) for t in (True, False)]" $Data
foreach ($name in "cifar-10-python.tar.gz", "cifar-100-python.tar.gz") {
  $cached = Join-Path $Cache $name
  if (-not (Test-Path $cached)) { Copy-Item (Join-Path $Data $name) $cached }
}
if ($SetupOnly) { Write-Host "Setup OK. Results would go to $Out"; Stop-Transcript | Out-Null; exit 0 }

# --- 6. run ------------------------------------------------------------------------------------------
# Windows PowerShell turns any stderr line of a redirected native command (e.g. a Python warning)
# into a terminating error under "Stop", so the run phase checks exit codes instead.
$ErrorActionPreference = "Continue"
$env:PCDKD_OUT = $Out
$env:PCDKD_WINDOW = "aus"
$select = @("--profile", $Profile)
if ($Family) { $select += @("--family") + $Family.Split(" ") }
$py = @("run", "--no-sync", "python", "scripts/run_suite.py")

# Claims left by a previous session are stale (DeepFreeze or a closed window ended it): clear once,
# before any worker starts.
uv run --no-sync python scripts/run_suite.py clear-claims @select
$args_ = $py + @("run") + $select + @("--queue", "--window", "aus", "--workers", $Workers,
                                      "--data-root", "`"$Data`"")
$procs = foreach ($i in 1..$Procs) {
  Start-Process -PassThru -NoNewWindow -FilePath "uv" -ArgumentList $args_ `
    -RedirectStandardOutput (Join-Path $Logs "worker$i-$stamp.log") `
    -RedirectStandardError (Join-Path $Logs "worker$i-$stamp.err")
  Start-Sleep -Seconds 20   # stagger start-up so the first teachers are claimed in order
}
Write-Host "$Procs workers running; logs in $Logs. Disconnect RDP (do not sign out) to leave."
$procs | Wait-Process
Write-Host "Session finished. Results: $Out  Logs: $Logs"
Stop-Transcript | Out-Null
