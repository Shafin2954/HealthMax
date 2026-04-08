param(
    [string]$PythonExe = "C:\Users\user\anaconda3\envs\healthmax312\python.exe",
    [string]$BindHost = "127.0.0.1",
    [int]$BindPort = 8000
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$hfHome = Join-Path $repoRoot ".cache\huggingface"

New-Item -ItemType Directory -Force -Path $hfHome | Out-Null

$env:HF_HOME = $hfHome
$env:TRANSFORMERS_CACHE = $hfHome

Set-Location $repoRoot
& $PythonExe -m uvicorn backend.main:app --host $BindHost --port $BindPort
