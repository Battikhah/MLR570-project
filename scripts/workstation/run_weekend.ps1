<#
Run the PCD suite on the AUS GPU workstation (ESB-1043-20) for one access window.

Rules this script follows:
  * Access only Thursday 18:00 - Monday 08:00 (UAE time). The Python runner refuses to start
    outside the window and pauses with a checkpoint 20 minutes before it closes.
  * DeepFreeze wipes the local disk on any restart. The code comes from git (this clone is
    pulled to the latest commit each session and re-cloned after a wipe). Runs write to the local
    disk, and rclone copies results, teacher weights, run checkpoints and logs to Google Drive
    (My Drive/pcd-suite) every -SyncMinutes, so a crash or wipe loses at most that much. At the
    start of a session after a wipe (or with -Restore) the Drive copy is pulled back, so runs resume from their last checkpoint.

What it runs: the whole suite (all 19 families of the chosen profile) as one queue shared by
-Procs processes on the GPU. It follows the explicit weekend plan in
src/pcdkd/experiments/suite.py (WEEKEND_PLAN), then appends all deferred work in its legacy order.
Each teacher is trained just before the first run that needs it. Rerunning the same command next
window continues where it stopped.

Usage, in PowerShell on the workstation (connected over GlobalProtect + RDP). Once per wipe:
  git clone https://github.com/Battikhah/MLR570-project "$env:LOCALAPPDATA\pcd-suite\code"
Then:
  cd "$env:LOCALAPPDATA\pcd-suite\code\scripts\workstation"
  powershell -ExecutionPolicy Bypass -File run_weekend.ps1 -SetupOnly        # test access only
  powershell -ExecutionPolicy Bypass -File run_weekend.ps1                   # whole suite, 3 procs
  powershell -ExecutionPolicy Bypass -File run_weekend.ps1 -Procs 4 -Family "label-noise loss-scale"
  powershell -ExecutionPolicy Bypass -File run_weekend.ps1 -Restore         # force the Drive restore
The Drive restore runs only when the local results are gone (after a reboot wipe); otherwise it is skipped.
The first run after a wipe opens a browser to sign in to Google for rclone.

Keep this PowerShell window open: it does the Drive uploads. When leaving, DISCONNECT the RDP
session (close the RDP window); do not sign out, which kills the run.
#>
param(
  [string]$Profile = "screen",
  [int]$Procs = 3,              # processes sharing the GPU (ResNet-8 students barely load one GPU)
  [string]$Family = "",         # optional subset, space-separated; default: every family
  [int]$Workers = 3,            # data-loading workers per process
  [int]$SyncMinutes = 5,        # how often results and checkpoints are copied to Google Drive
  [string]$Remote = "gdrive",   # rclone remote name
  [string]$Data = "",           # CIFAR folder (e.g. on OneDrive); default: local, re-downloaded after a wipe
  [switch]$SetupOnly,
  [switch]$Restore              # force the Drive restore even if local results survived
)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"   # Windows defaults to cp1252, which cannot write the Greek letters in reports and logs
$env:PYTHONUNBUFFERED = "1"   # worker logs are written line by line (the dashboard reads them live)

# --- where things live (all local, wiped by DeepFreeze; Google Drive holds the copy) ---------------
$Work  = Join-Path $env:LOCALAPPDATA "pcd-suite"
$Out   = Join-Path $Work "outputs"      # results, teachers, checkpoints
$Logs  = Join-Path $Work "logs"
if (-not $Data) { $Data = Join-Path $Work "data" }
$Code  = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path   # this git clone
$Dest  = "${Remote}:pcd-suite"
New-Item -ItemType Directory -Force $Out, $Logs, $Data | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
Start-Transcript -Path (Join-Path $Logs "session-$stamp.log") | Out-Null

# --- 1. uv, per user (no admin rights needed) ----------------------------------------------------
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# --- 2. rclone (portable exe, no admin) and the Google Drive remote ---------------------------------
$RcloneVer = "v1.68.2"
$rclone = Join-Path $Work "rclone-$RcloneVer-windows-amd64\rclone.exe"
if (-not (Test-Path $rclone)) {
  $zip = Join-Path $Work "rclone.zip"
  Invoke-WebRequest "https://downloads.rclone.org/$RcloneVer/rclone-$RcloneVer-windows-amd64.zip" -OutFile $zip
  Expand-Archive $zip $Work -Force
}
if (-not ((& $rclone listremotes) -contains "${Remote}:")) {
  # drive.file: rclone can only see the files it creates, not the rest of your Drive.
  Write-Host "A browser will open: sign in to the Google account whose Drive should hold the results."
  & $rclone config create $Remote drive scope=drive.file
  if ($LASTEXITCODE -ne 0) { throw "Google Drive sign-in for rclone failed." }
}
& $rclone mkdir "$Dest/outputs"
if ($LASTEXITCODE -ne 0) { throw "Cannot write to Google Drive ($Dest)." }
function Push-Drive {
  # copy, never sync: nothing on Drive is ever deleted. Claims are per session; *.tmp are half-written.
  & $rclone copy $Out "$Dest/outputs" --exclude "claim" --exclude "*.claim" --exclude "*.tmp" --fast-list -q
  & $rclone copy $Logs "$Dest/logs" -q
}

# --- 3. pull back what earlier sessions saved, so runs resume ----------------------------------------
# Only needed after a wipe (reboot): without one, local results survive and are newer than Drive.
# -Restore forces it. --ignore-existing: never overwrite a local file, and no per-file modtime compare.
$haveLocal = (Test-Path (Join-Path $Out $Profile)) -and
  [bool](Get-ChildItem (Join-Path $Out $Profile) -Recurse -Filter result.json -File -ErrorAction SilentlyContinue | Select-Object -First 1)
if ($haveLocal -and -not $Restore) {
  Write-Host "Local results present (no wipe since the last session): skipping the Drive restore. Use -Restore to force it."
} else {
  Write-Host "Restoring earlier results from Google Drive; progress every 30 s (a large Drive can take several minutes) ..."
  & $rclone copy "$Dest/outputs" $Out --ignore-existing --exclude "claim" --exclude "*.claim" --exclude "error.json" --exclude "*.error.json" --exclude "*.tmp" --fast-list --checkers 16 --transfers 8 --stats 30s --stats-one-line --stats-log-level NOTICE
  if ($LASTEXITCODE -ne 0) { throw "Could not restore earlier results from Google Drive." }
}

# --- 4. latest code from git -----------------------------------------------------------------------
Set-Location $Code
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw "git pull failed in $Code; fix it (or delete the folder and clone again)." }
Write-Host "Code: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"   # also saved in each result.json

# --- 5. environment: the locked packages, then a CUDA build of torch -------------------------------
# PyPI's Windows torch wheels are CPU-only. The driver (571.59) supports CUDA <= 12.8, and the
# locked torch 2.15 has only CUDA 13.2 builds for Windows, so install the newest CUDA 12.6 build:
# torch 2.14.1 with torchvision 0.29.1. Every later command uses `uv run --no-sync` so uv does not
# put the locked CPU build back.
uv sync --frozen
uv pip install --reinstall "torch==2.14.1" "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw "installing the CUDA build of torch failed" }
uv run --no-sync python -c "import sys, torch; ok = torch.cuda.is_available(); print('CUDA:', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA is not available to torch; check the GPU driver before running anything." }

# --- 6. sanity tests (CPU, synthetic data) ----------------------------------------------------------
uv run --no-sync pytest -q -m "not slow and not requires_data"
if ($LASTEXITCODE -ne 0) { throw "tests failed; not starting GPU runs" }

# --- 7. CIFAR-10/100 (skipped when already downloaded this session) ---------------------------------
uv run --no-sync python -c "from torchvision.datasets import CIFAR10, CIFAR100; import sys; [c(sys.argv[1], train=t, download=True) for c in (CIFAR10, CIFAR100) for t in (True, False)]" $Data
if ($LASTEXITCODE -ne 0) { throw "CIFAR download failed" }

# --- 7b. CRD's pre-trained CIFAR-100 teachers (mdistiller release, for lit-p1), once per -Data -----
# The suite reads them as mdistiller/cifar_teachers/... under --data-root, so nothing else is passed.
$Md = Join-Path $Data "mdistiller"
if (-not (Test-Path (Join-Path $Md "cifar_teachers\resnet32x4_vanilla\ckpt_epoch_240.pth"))) {
  New-Item -ItemType Directory -Force $Md | Out-Null
  $MdTar = Join-Path $Md "cifar_teachers.tar"
  $MdSha = "E7C8B2F7D449AF08E1222663E845AD23CA4A2044735B980740D86A96088DE52C"
  if (-not (Test-Path $MdTar) -or (Get-Item $MdTar).Length -ne 364390400) {
    $ProgressPreference = "SilentlyContinue"   # the progress bar makes Invoke-WebRequest very slow
    Invoke-WebRequest "https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar" -OutFile $MdTar
    $ProgressPreference = "Continue"
  }
  if ((Get-Item $MdTar).Length -ne 364390400 -or (Get-FileHash $MdTar -Algorithm SHA256).Hash -ne $MdSha) {
    Remove-Item $MdTar; throw "cifar_teachers.tar has the wrong size or SHA-256; deleted, rerun to retry."
  }
  tar -xf $MdTar -C $Md   # Windows 10+ ships bsdtar as tar.exe
  if ($LASTEXITCODE -ne 0) { throw "extracting $MdTar failed" }
}
if ($SetupOnly) {
  Write-Host "Setup OK. Results will be copied to Google Drive: My Drive/pcd-suite (this log is there now)."
  Stop-Transcript | Out-Null; Push-Drive; exit 0
}

# --- 8. run ------------------------------------------------------------------------------------------
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
# Live dashboard (read-only) at http://127.0.0.1:8765; if it fails, the workers still run.
$dashboard = $null
try {
  $dashboard = Start-Process -PassThru -WindowStyle Hidden -FilePath "uv" `
    -ArgumentList @("run", "--no-sync", "python", "scripts/dashboard.py", "--out", "`"$Out`"",
                    "--logs", "`"$Logs`"", "--profile", $Profile) `
    -RedirectStandardOutput (Join-Path $Logs "dashboard-$stamp.out") `
    -RedirectStandardError (Join-Path $Logs "dashboard-$stamp.err")
  Write-Host "Dashboard: http://127.0.0.1:8765 (opens in the browser in a few seconds)"
  Start-Job { Start-Sleep -Seconds 15; Start-Process "http://127.0.0.1:8765" } | Out-Null
} catch { Write-Host "Dashboard did not start ($_); the runs continue without it." }
$running = foreach ($i in 1..$Procs) {
  Start-Process -PassThru -NoNewWindow -FilePath "uv" -ArgumentList $args_ `
    -RedirectStandardOutput (Join-Path $Logs "worker$i-$stamp.log") `
    -RedirectStandardError (Join-Path $Logs "worker$i-$stamp.err")
  Start-Sleep -Seconds 20   # stagger start-up so the first teachers are claimed in order
}
Write-Host "$Procs workers running; logs in $Logs. Keep this window open (it uploads to Google Drive)."
Write-Host "Disconnect RDP (do not sign out) to leave."
while ($running | Where-Object { -not $_.HasExited }) {
  $running | Wait-Process -Timeout ($SyncMinutes * 60) -ErrorAction SilentlyContinue
  Push-Drive
}
Write-Host "Session finished. Results: Google Drive My Drive/pcd-suite/outputs"
if ($dashboard) { taskkill /T /F /PID $dashboard.Id 2>$null | Out-Null }   # uv and its python child
Stop-Transcript | Out-Null
Push-Drive
