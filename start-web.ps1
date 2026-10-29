param([switch]$NoBrowser, [UInt16]$AppPortNumber = 0)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$appPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $appPython)) {
    & (Join-Path $PSScriptRoot 'setup-app.ps1')
    return
}
& $appPython -B -c 'import sys; from mosaic_refiner.runtime_settings import selection_ready; sys.exit(0 if selection_ready() else 1)'
if ($LASTEXITCODE -ne 0) {
    & $appPython -B -m mosaic_refiner setup-gui
    return
}
# Only environments created by this installer have an app-owned launcher.
# Existing ComfyUI is started through the user's usual entry point.
& $appPython -B -c "from mosaic_refiner.runtime_settings import RUNTIME; from mosaic_refiner.setup_environment import state_directory,start_managed; import json; p=state_directory(RUNTIME)/'state.json'; start_managed(RUNTIME) if p.is_file() and json.loads(p.read_text(encoding='utf-8')).get('managed_new') else None"
if ($LASTEXITCODE -ne 0) { throw '選択した推論環境を起動できません。初回セットアップの記録を確認してください。' }
$appLogDirectory = Join-Path $env:LOCALAPPDATA 'MosaicRefiner/logs'
New-Item -ItemType Directory -Path $appLogDirectory -Force | Out-Null
$appRunName = 'web-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$appPort = $null
$appNativeHost = Join-Path $PSScriptRoot 'mosaic_refiner/_native/mosaic-refiner.exe'
if (-not (Test-Path -LiteralPath $appNativeHost)) { throw 'Rustアプリを scripts/build_native.py で構築してください。起動時の自動構築は行いません。' }
$appRevision = 'rust-host-v2-model-auth'
$appBuild = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'mosaic_refiner/_native/host-build.json') | ConvertFrom-Json
$appToken = $null
$appUrl = $null
$appMismatch = $false
# Reuse only a private startup URL in this Windows user's LOCALAPPDATA logs.
# The public HTML never supplies an operation key.
foreach ($appPreviousLog in @(Get-ChildItem -LiteralPath $appLogDirectory -Filter 'web-*.stdout.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 20)) {
    $appLine = Get-Content -LiteralPath $appPreviousLog.FullName -TotalCount 1 -Encoding UTF8 -ErrorAction SilentlyContinue
    if ($appLine -notmatch '^WebGUI: (http://127\.0\.0\.1:(\d+)/#token=([A-Za-z0-9_-]+))') { continue }
    $appPreviousUrl = $Matches[1]; $appPreviousPort = [int]$Matches[2]; $appPreviousToken = $Matches[3]
    try {
        $appStatus = Invoke-RestMethod "http://127.0.0.1:$appPreviousPort/api/status" -Headers @{'X-Mosaic-Token'=$appPreviousToken} -TimeoutSec 2
        if ($appStatus.pipeline_revision -eq $appRevision -and $appStatus.application_build -eq $appBuild.source_sha256 -and $appStatus.application_root -eq $PSScriptRoot) {
            $appPort = $appPreviousPort; $appToken = $appPreviousToken; $appUrl = $appPreviousUrl; break
        }
        if ($appStatus.application_engine -eq 'rust') { $appMismatch = $true }
    } catch { }
}
if ($null -eq $appPort) {
    foreach ($appCandidatePort in @(8765, 8766)) {
        $appListener = Get-NetTCPConnection -LocalPort $appCandidatePort -State Listen -ErrorAction SilentlyContinue
        if ($appListener) {
            # An old unauthenticated version may be running. Never reuse it.
            try {
                $appOldStatus = Invoke-RestMethod "http://127.0.0.1:$appCandidatePort/api/status" -TimeoutSec 1
                if ($appOldStatus.application_engine -eq 'rust') { $appMismatch = $true }
            } catch { }
        }
    }
    if ($appMismatch) { throw '別の版・場所のMosaicRefinerが実行中です。その画面で処理を停止・保存し、アプリを終了してから起動してください。' }
    # Port 0 lets the OS select a free UI port; ComfyUI's selected API is unchanged.
    $appPort = $AppPortNumber
    $appStdout = Join-Path $appLogDirectory "$appRunName.stdout.log"
    Start-Process -FilePath $appNativeHost -ArgumentList 'web', '--port', $appPort -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
        -RedirectStandardOutput $appStdout `
        -RedirectStandardError (Join-Path $appLogDirectory "$appRunName.stderr.log")
}
$appReady = $false
for ($appAttempt = 0; $appAttempt -lt 40; $appAttempt++) {
    if (-not $appToken -and $appStdout -and (Test-Path -LiteralPath $appStdout)) {
        $appLine = Get-Content -LiteralPath $appStdout -TotalCount 1 -Encoding UTF8 -ErrorAction SilentlyContinue
        if ($appLine -match '^WebGUI: (http://127\.0\.0\.1:(\d+)/#token=([A-Za-z0-9_-]+))' -and ($appPort -eq 0 -or [int]$Matches[2] -eq $appPort)) {
            $appUrl = $Matches[1]; $appToken = $Matches[3]; $appPort = [int]$Matches[2]
        }
    }
    if ($appToken) {
        try {
            $appStatus = Invoke-RestMethod "http://127.0.0.1:$appPort/api/status" -Headers @{'X-Mosaic-Token'=$appToken} -TimeoutSec 1
            if ($appStatus.pipeline_revision -eq $appRevision -and $appStatus.application_build -eq $appBuild.source_sha256 -and $appStatus.application_root -eq $PSScriptRoot) {
                $appReady = $true; break
            }
        } catch { }
    }
    Start-Sleep -Milliseconds 300
}
if (-not $appReady) { throw "WebGUIが起動しませんでした。ログ: $appLogDirectory" }
# Intermediates remain until explicit history cleanup.
if (-not $NoBrowser) { Start-Process $appUrl }
