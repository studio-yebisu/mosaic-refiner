param([switch]$Setup)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $Setup) {
    & (Join-Path $PSScriptRoot 'start-web.ps1')
    return
}
$appPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $appPython)) {
    & (Join-Path $PSScriptRoot 'setup-app.ps1')
    return
}
if (-not $Setup) {
    & $appPython -B -c 'import sys; from mosaic_refiner.runtime_settings import selection_ready; sys.exit(0 if selection_ready() else 1)'
    if ($LASTEXITCODE -ne 0) { $Setup = $true }
}
$appModule = if ($Setup) { 'mosaic_refiner.setup_gui' } else { 'mosaic_refiner.gui' }
if (-not $Setup) {
    & $appPython -B -c "from mosaic_refiner.runtime_settings import RUNTIME; from mosaic_refiner.setup_environment import state_directory,start_managed; import json; p=state_directory(RUNTIME)/'state.json'; start_managed(RUNTIME) if p.is_file() and json.loads(p.read_text(encoding='utf-8')).get('managed_new') else None"
    if ($LASTEXITCODE -ne 0) { throw '選択した推論環境を起動できません。初回セットアップの記録を確認してください。' }
}
$appLogDirectory = Join-Path $env:LOCALAPPDATA 'MosaicRefiner/logs'
New-Item -ItemType Directory -Path $appLogDirectory -Force | Out-Null
$appRunName = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
Start-Process -FilePath $appPython -ArgumentList '-B', '-m', $appModule -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $appLogDirectory "$appRunName.stdout.log") `
    -RedirectStandardError (Join-Path $appLogDirectory "$appRunName.stderr.log")
if (-not $Setup) { & (Join-Path $PSScriptRoot 'start-cleanup.ps1') }
