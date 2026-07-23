# Windows helper: install build deps and package Pong + Tetris.
#   .\build_demos.ps1
#   .\build_demos.ps1 pong
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (Test-Path "$Root\.venv\Scripts\python.exe") {
    $Py = "$Root\.venv\Scripts\python.exe"
} else {
    $Py = "python"
}

Write-Host "Using: $Py"
& $Py -m pip install -q -r requirements-build.txt
& $Py build_demos.py @args
