param(
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

$mode = "--onedir"
if ($OneFile) { $mode = "--onefile" }

python -m PyInstaller --noconfirm --clean --name CaroAI --windowed $mode main.py

if ($OneFile) {
    Write-Host "Built: dist\\CaroAI.exe"
} else {
    Write-Host "Built: dist\\CaroAI\\CaroAI.exe"
}
