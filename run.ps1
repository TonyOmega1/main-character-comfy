# MC launcher (Windows PowerShell).
#   .\run.ps1          real renders (needs ComfyUI running on :8188)
#   .\run.ps1 -Mock    UI demo without a GPU
#   .\run.ps1 -Check   just check: is ComfyUI up, and which model files are still missing for your GPU tier
#   .\run.ps1 -Agent  also serve the agent connector (MCP) at http://127.0.0.1:8765/mcp
#   .\run.ps1 -Agent -Remote   same, bearer token required (for tailscale serve / a cloudflare tunnel)
#   .\run.ps1 -StartComfy C:\AI\ComfyUI_windows_portable
#              also start ComfyUI if it isn't running, ALWAYS with --reserve-vram 1.5 --max-upload-size 500
#              (MC's defaults: VRAM kept free for the desktop; uploads over ComfyUI's 100 MB default fail with 413).
#              If the folder has run_nvidia_gpu_mc.bat (those flags), MC starts that .bat.
#              -ReserveVram 2 for streaming (OBS/NVENC), -MaxUploadMB 1024 for very long clips,
#              -ComfyLauncher <script> to start your own launcher script instead.
#   .\run.ps1 -Python "C:\AI\ComfyUI_windows_portable\python_embeded\python.exe"   (not recommended; use a normal Python 3.10+)
param(
    [switch]$Mock,
    [switch]$Check,
    [switch]$Agent,
    [switch]$Remote,
    [string]$Python = "py",
    [string]$ComfyUI = "http://127.0.0.1:8188",
    [int]$ProxyPort = 8189,
    [string]$StartComfy = "",
    [double]$MaxUploadMB = 500,
    [double]$ReserveVram = 1.5,
    [string]$ComfyLauncher = ""
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$FirstRun = -not (Test-Path ".venv")
if ($FirstRun) {
    if ($Python -eq "py" -and -not (Get-Command py -ErrorAction SilentlyContinue)) {
        Write-Host "Python 3.10+ is needed for the MC app (separate from ComfyUI's embedded Python). Install it with:" -ForegroundColor Yellow
        Write-Host "  winget install -e --id Python.Python.3.12" -ForegroundColor Yellow
        Write-Host "then open a new PowerShell and run .\run.ps1 again."
        exit 1
    }
    Write-Host "Creating virtual environment..."
    if ($Python -eq "py") { & py -3 -m venv .venv } else { & $Python -m venv .venv }
    & .\.venv\Scripts\python.exe -m pip install -q --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -q -r requirements.txt
}
$env:COMFY_BASE_URL = "http://127.0.0.1:$ProxyPort"
$env:COMFYUI_URL = $ComfyUI

function Test-Comfy {
    try { Invoke-RestMethod -Uri "$ComfyUI/system_stats" -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}
if ($StartComfy -and -not $Mock) {
    if (Test-Comfy) {
        Write-Host "ComfyUI is already running; its flags:"
        & .\.venv\Scripts\python.exe -m main_character_comfy.comfy_launch --flags --url $ComfyUI
        Write-Host "(restart ComfyUI to change -MaxUploadMB / -ReserveVram)"
    } else {
        if ($ComfyLauncher) { $env:MC_COMFY_LAUNCHER = $ComfyLauncher }
        $la = @("-m", "main_character_comfy.comfy_launch", "--root", $StartComfy,
                "--max-upload-size", "$MaxUploadMB", "--reserve-vram", "$ReserveVram")
        & .\.venv\Scripts\python.exe @la
        Write-Host "Waiting for ComfyUI at $ComfyUI ..."
        for ($i = 0; $i -lt 180 -and -not (Test-Comfy); $i++) { Start-Sleep -Seconds 1 }
        if (-not (Test-Comfy)) { Write-Warning "ComfyUI isn't answering yet; check its console window." }
    }
}

if ($Check -or $FirstRun) {
    Write-Host "Checking ComfyUI and your model files..."
    & .\.venv\Scripts\python.exe -m main_character_comfy.doctor --live $ComfyUI
    if ($Check) { exit $LASTEXITCODE }
}

if ($Mock) {
    & .\.venv\Scripts\python.exe app.py --backend mock
    exit $LASTEXITCODE
}

function Test-Proxy {
    try { Invoke-RestMethod -Uri "$env:COMFY_BASE_URL/api/v2/health" -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

if (-not (Test-Proxy)) {
    Write-Host "Starting comfy-api-proxy ($ComfyUI -> $env:COMFY_BASE_URL)..."
    Start-Process -FilePath ".\.venv\Scripts\comfy-api-proxy.exe" `
        -ArgumentList @("run", "--comfyui", $ComfyUI, "--port", "$ProxyPort") `
        -WindowStyle Minimized
    for ($i = 0; $i -lt 30 -and -not (Test-Proxy); $i++) { Start-Sleep -Milliseconds 500 }
    if (-not (Test-Proxy)) { Write-Warning "Proxy not answering yet. Is ComfyUI running at $ComfyUI ?" }
}
$extra = @()
if ($Agent) { $extra += "--agent" }
if ($Remote) { $extra += "--remote" }
& .\.venv\Scripts\python.exe app.py --backend comfy @extra @args
