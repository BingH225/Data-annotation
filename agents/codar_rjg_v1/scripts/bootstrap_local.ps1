param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Write-Host "[bootstrap_local] projectRoot=$projectRoot"

Push-Location $projectRoot
try {
  & $PythonExe -m venv .venv
  . .\.venv\Scripts\Activate.ps1
  python -m pip install -U pip
  python -m pip install -e .
  if (-not (Test-Path config\runtime.yaml)) {
    Copy-Item config\runtime.template.yaml config\runtime.yaml
    Write-Host "[bootstrap_local] created config/runtime.yaml from template"
  }
  Write-Host "[bootstrap_local] done"
}
finally {
  Pop-Location
}
