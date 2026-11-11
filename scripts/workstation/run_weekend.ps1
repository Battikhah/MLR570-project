<#
Run the PCD suite on the AUS GPU workstation (ESB-1043-20) for one access window.

Rules this script follows:
  * Access only Thursday 18:00 - Monday 08:00 (UAE time). The Python runner refuses to start
    outside the window and pauses with a checkpoint 20 minutes before it closes.
  * DeepFreeze wipes the local disk on any restart. Everything that matters (results, teacher
    weights, run checkpoints, logs, the CIFAR archives) is written to OneDrive; the local copy of
    the code, Python environment and datasets is rebuilt each session.

Usage, in PowerShell on the workstation (connected over GlobalProtect + RDP):
  powershell -ExecutionPolicy Bypass -File run_weekend.ps1 -Profile screen -Tier "0" -Stage teachers
  powershell -ExecutionPolicy Bypass -File run_weekend.ps1 -Profile screen -Tier "0 1" -Shards 3
  powershell -ExecutionPolicy Bypass -File run_weekend.ps1 -SetupOnly        # test access only

When leaving, DISCONNECT the RDP session (close the window); do not sign out, which kills the run.
#>
param(
  [string]$Profile = "screen",
  [string]$Tier = "0",
  [ValidateSet("all", "teachers", "students")][string]$Stage = "all",
  [int]$Shards = 1,
  [int]$Workers = 4,
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
$common = @("scripts/run_suite.py", "run", "--profile", $Profile, "--tier") + $Tier.Split(" ") +
          @("--data-root", $Data, "--workers", $Workers, "--window", "aus")
if ($Shards -le 1) {
  uv run --no-sync python @common --stage $Stage 2>&1 | Tee-Object -FilePath (Join-Path $Logs "run-$stamp.log")
} else {
  # Teachers once, then one process per shard of students (small students do not fill one GPU).
  if ($Stage -ne "students") {
    uv run --no-sync python @common --stage teachers 2>&1 | Tee-Object -FilePath (Join-Path $Logs "teachers-$stamp.log")
  }
  if ($Stage -ne "teachers") {
    $procs = foreach ($i in 0..($Shards - 1)) {
      Start-Process -PassThru -NoNewWindow -FilePath "uv" `
        -ArgumentList (@("run", "--no-sync", "python") + $common + @("--stage", "students", "--shard", "$i/$Shards")) `
        -RedirectStandardOutput (Join-Path $Logs "shard$i-$stamp.log") `
        -RedirectStandardError (Join-Path $Logs "shard$i-$stamp.err")
    }
    $procs | Wait-Process
  }
}
Write-Host "Session finished. Results: $Out  Logs: $Logs"
Stop-Transcript | Out-Null
