# OHH-46 — hardware-in-the-loop runner for the OmniBot + Go2 bench (Windows).
$ErrorActionPreference = "Stop"

$env:OHHO_HIL = "1"
if (-not $env:OHHO_OMNIBOT_PORT) { $env:OHHO_OMNIBOT_PORT = "COM3" }
if (-not $env:OHHO_OMNIBOT_ARM) { $env:OHHO_OMNIBOT_ARM = "COM4" }
if (-not $env:OHHO_GO2_IFACE) { $env:OHHO_GO2_IFACE = "eth0" }

Write-Host "OHHO_HIL=1"
Write-Host "  OHHO_OMNIBOT_PORT=$($env:OHHO_OMNIBOT_PORT)"
Write-Host "  OHHO_OMNIBOT_ARM=$($env:OHHO_OMNIBOT_ARM)"
Write-Host "  OHHO_GO2_IFACE=$($env:OHHO_GO2_IFACE)"
Write-Host ""
Write-Host "Install extras as needed: pip install -e '.[serial,arm,unitree]'"
Write-Host ""

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
python -m unittest discover -s tests/hil -v
