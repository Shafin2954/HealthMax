param(
    [string]$NodeDir = "C:\Users\user\anaconda3\envs\healthmax312",
    [string]$BindHost = "127.0.0.1",
    [int]$BindPort = 5173
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $repoRoot "healthmax-ai-assistant"
$npmCmd = Join-Path $NodeDir "npm.cmd"

$env:Path = "$NodeDir;$env:Path"

Set-Location $frontendRoot
& $npmCmd run dev -- --host $BindHost --port $BindPort
