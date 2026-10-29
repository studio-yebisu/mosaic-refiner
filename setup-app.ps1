param([switch]$PlanOnly, [switch]$AcceptTerms, [switch]$NoGui)
$ErrorActionPreference = 'Stop'
$appRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$appRecipe = Get-Content -LiteralPath (Join-Path $appRoot 'mosaic_refiner/setup_assets/recipe.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$appHome = Join-Path $env:LOCALAPPDATA 'MosaicRefiner'
$appPythonHome = Join-Path $appHome 'python313'
$appBasePython = Join-Path $appPythonHome 'python.exe'
$appPython = Join-Path $appRoot '.venv/Scripts/python.exe'
$appCache = Join-Path $appHome 'bootstrap-downloads'
$appLog = Join-Path $appHome 'logs/bootstrap.log'
. (Join-Path $appRoot 'scripts/resolve-bootstrap.ps1')
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
if ($PlanOnly) {
    $appResolvedPython = Resolve-MosaicPython
} else { $appResolvedPython = $null }
$appPlan = [ordered]@{ app = $appRoot; python = $appBasePython; venv = $appPython;
    pythonResolution = $appResolvedPython; pythonSelection = 'latest final Windows installer in supported 3.13 series';
    vcRuntimeUrl = $appRecipe.vc_runtime.url; comfyCreated = $false; modelsDownloaded = $false;
    driverInstalled = $false; generationStarted = $false }
if ($PlanOnly) { $appPlan | ConvertTo-Json -Depth 4; return }
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -notin @('AMD64','x86')) {
    throw '自動導入はx64 Windows向けです。ARM版は対象外です。'
}
if (-not $AcceptTerms) {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    $appForm = New-Object Windows.Forms.Form
    $appForm.Text = 'Mosaic Refiner — アプリを準備'
    $appForm.Size = New-Object Drawing.Size(660,430)
    $appForm.StartPosition = 'CenterScreen'
    $appInfo = New-Object Windows.Forms.TextBox
    $appInfo.Multiline = $true; $appInfo.ReadOnly = $true; $appInfo.ScrollBars = 'Vertical'
    $appInfo.SetBounds(20,20,600,240)
    $appInfo.Text = "ジョークグッズとしての実験アプリです。元情報の復元は保証しません。`r`n`r`nPython・アプリ用ライブラリ・不足するVisual C++ランタイムを準備して、導入画面を開きます。Rustの構築には別途Rust/Cargo 1.89以上とC++ Build Tools / Windows SDKが必要です。`r`n`r`n既存ComfyUIがあればその環境を選択します。推奨はVRAM 24GB / RAM 64GBです。16GB VRAMは動く可能性がありますが未実測、12〜23GBは未検証です。Windows 10 22H2 / 11 x64、NVIDIA Ampere以降・R580以降が必要です。`r`n`r`nアプリ: AGPL-3.0-only。モデルの許諾は別です。READMEとdocs/use-conditions.mdを確認してください。`r`n`r`nPython: https://www.python.org/about/legal/`r`n`r`nMicrosoft: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/`r`n`r`nドライバー更新・モデル取得・動画処理はこの画面では行いません。Microsoftランタイムの導入で権限画面が出る場合があります。"
    $appConsent = New-Object Windows.Forms.CheckBox
    $appConsent.Text = '配布元の利用条件を確認しました'; $appConsent.SetBounds(20,275,570,30)
    $appButton = New-Object Windows.Forms.Button
    $appButton.Text = 'アプリを準備してセットアップへ'; $appButton.Enabled = $false; $appButton.SetBounds(260,325,350,35)
    $appConsent.Add_CheckedChanged({ $appButton.Enabled = $appConsent.Checked })
    $appButton.DialogResult = [Windows.Forms.DialogResult]::OK
    $appForm.Controls.AddRange(@($appInfo,$appConsent,$appButton))
    if ($appForm.ShowDialog() -ne [Windows.Forms.DialogResult]::OK) { return }
}
New-Item -ItemType Directory -Path $appCache,(Split-Path -Parent $appLog) -Force | Out-Null
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
try {
    # A newer compatible installed VC runtime is reused. No SDK/driver install.
    $appVc = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue
    $appVcVersion = if ($appVc -and $appVc.Installed -eq 1) { [version]($appVc.Version -replace '^v','') } else { [version]'0.0' }
    if ($appVcVersion -lt [version]$appRecipe.vc_runtime.minimum_version) {
        Write-Host 'Microsoft Visual C++ランタイムを準備しています…'
        $appVcFile = Get-AppInstaller $appRecipe.vc_runtime 'vc-redist-x64.exe' 'Microsoft Corporation'
        $appVcProcess = Start-Process -FilePath $appVcFile -ArgumentList '/install','/quiet','/norestart' -WindowStyle Hidden -Wait -PassThru
        if ($appVcProcess.ExitCode -eq 3010) { throw 'Visual C++ランタイムを導入しました。Windowsを再起動して、このセットアップをもう一度実行してください。' }
        if ($appVcProcess.ExitCode -ne 0) { throw "Visual C++導入に失敗しました（終了値$($appVcProcess.ExitCode)）。権限画面とMicrosoftの案内を確認してください。" }
    }
    $appPython = Initialize-MosaicAppVenv $appRoot $appHome
    Write-Host 'アプリ用ライブラリを準備しています…'
    # Update only the application's installer, before it resolves other wheels.
    Invoke-AppPip $appPython @('install','--only-binary=:all:','--upgrade','pip','--index-url','https://pypi.org/simple')
    # Editable source is local; third-party wheels only. The ComfyUI venv is separate.
    Invoke-AppPip $appPython @('install','--only-binary=:all:','-e',$appRoot,'--index-url','https://pypi.org/simple')
    Invoke-AppPip $appPython @('check')
    Write-Host 'アプリの準備ができました。次に推論環境を選びます。'
    if (-not $NoGui) { Invoke-AppPython $appPython @('-B','-m','mosaic_refiner','setup-gui') }
} catch {
    Write-Error $_
    exit 1
}
