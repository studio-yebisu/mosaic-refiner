# Functions only: dot-sourcing this file makes no network request.
function Get-MosaicOfficialText([string]$Uri) {
    if ($Uri -notmatch '^https://www\.python\.org/ftp/python/(?:[0-9./A-Za-z_-]+)?$') {
        throw 'Pythonの取得元は公式HTTPSへ限定します。'
    }
    $reply = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 30 -MaximumRedirection 0
    if ($reply.Content -is [byte[]]) { return [Text.Encoding]::UTF8.GetString($reply.Content) }
    return [string]$reply.Content
}

function Resolve-MosaicPython {
    # Use the newest final Windows installer in the supported 3.13 series.
    # A directory prepared before an installer is released is not a candidate.
    $listing = Get-MosaicOfficialText 'https://www.python.org/ftp/python/'
    $versions = @([regex]::Matches($listing, 'href="(3\.13\.[0-9]+)/"') |
        ForEach-Object { [version]$_.Groups[1].Value } | Sort-Object -Unique -Descending)
    foreach ($candidate in $versions) {
        $version = $candidate.ToString()
        $directory = 'https://www.python.org/ftp/python/' + $version + '/'
        $files = Get-MosaicOfficialText $directory
        $filename = 'python-' + $version + '-amd64.exe'
        if ($files -notmatch ('href="' + [regex]::Escape($filename) + '"')) { continue }
        $url = $directory + $filename
        $bundle = (Get-MosaicOfficialText ($url + '.sigstore')) | ConvertFrom-Json
        if ($bundle.messageSignature.messageDigest.algorithm -ne 'SHA2_256') {
            throw '公式PythonのSHA-256を確認できません。旧版へ戻さず停止します。'
        }
        $hashBytes = [Convert]::FromBase64String($bundle.messageSignature.messageDigest.digest)
        if ($hashBytes.Length -ne 32) { throw '公式Pythonのdigest形式が不正です。' }
        $hash = ([BitConverter]::ToString($hashBytes)).Replace('-','').ToLowerInvariant()
        $head = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 30 -MaximumRedirection 0
        $size = [long]$head.Headers['Content-Length']
        if ($size -le 0 -or $size -gt 100MB) { throw '公式Pythonの取得容量が不正です。' }
        return [pscustomobject]@{ id=('python-' + $version.Replace('.','-')); version=$version;
            url=$url; sha256=$hash; size_bytes=$size; digest_source=($url + '.sigstore');
            resolved_at=[DateTime]::UtcNow.ToString('o'); sigstore_signature_verified=$false }
    }
    throw '対応する公式Windows Python installerがありません。旧固定版へ戻さず停止します。'
}

function Test-MosaicInstallerSignature($Signature, [string]$Publisher) {
    if ($Publisher -cnotin @('Python Software Foundation', 'Microsoft Corporation') -or
        $Signature.Status -ne 'Valid' -or $null -eq $Signature.SignerCertificate) { return $false }
    $name = $Signature.SignerCertificate.GetNameInfo(
        [Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false)
    return [string]::Equals($name, $Publisher, [StringComparison]::Ordinal)
}

function Get-MosaicFileSha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hash.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $hash.Dispose(); $stream.Dispose() }
}

function Get-AppInstaller($Spec, [string]$Name, [string]$Publisher) {
    $file = Join-Path $appCache $Name
    if (-not (Test-Path -LiteralPath $file)) {
        $partial = Join-Path $appCache ($Name + '.' + [guid]::NewGuid().ToString('N') + '.partial')
        Invoke-WebRequest -Uri $Spec.url -OutFile $partial -UseBasicParsing -TimeoutSec 60
        if ((Get-Item -LiteralPath $partial).Length -ne $Spec.size_bytes -or
            (Get-MosaicFileSha256 $partial) -cne $Spec.sha256) {
            throw "取得ファイルの容量/hashが一致しません。保持先: $partial"
        }
        if (-not (Test-MosaicInstallerSignature (Get-AuthenticodeSignature -LiteralPath $partial) $Publisher)) {
            throw "installerの発行元署名を確認できません。保持先: $partial"
        }
        Move-Item -LiteralPath $partial -Destination $file
    }
    if ((Get-Item -LiteralPath $file).Length -ne $Spec.size_bytes -or
        (Get-MosaicFileSha256 $file) -cne $Spec.sha256) {
        throw "取得済みinstallerのhashが異なります: $file"
    }
    if (-not (Test-MosaicInstallerSignature (Get-AuthenticodeSignature -LiteralPath $file) $Publisher)) {
        throw "installerの発行元署名を確認できません: $file"
    }
    return $file
}

function Invoke-AppPython([string]$Exe, [object[]]$Arguments) {
    & $Exe @Arguments >> $appLog 2>&1
    if ($LASTEXITCODE -ne 0) { throw "アプリ環境の準備に失敗しました。ログ: $appLog" }
}

function Invoke-AppPip([string]$Exe, [object[]]$Arguments) {
    $saved = @{}
    foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
        if ($entry.Key -like 'PIP_*') { $saved[$entry.Key] = $entry.Value }
    }
    try {
        foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $null, 'Process') }
        # pip compares this value to os.devnull before loading even global/site
        # config. Windows' os.devnull is lower-case; NUL is not the same string.
        [Environment]::SetEnvironmentVariable('PIP_CONFIG_FILE', 'nul', 'Process')
        $pipArguments = @('-B', '-m', 'pip', '--isolated') + $Arguments
        if ($Arguments.Count -gt 0 -and $Arguments[0] -eq 'install') {
            $index = [Array]::IndexOf($Arguments, '--index-url')
            if ($index -ge 0) {
                if ($index + 1 -ge $Arguments.Count -or $Arguments[$index + 1] -cne 'https://pypi.org/simple') {
                    throw 'アプリ依存の取得元は公式PyPIへ限定します。'
                }
            } else { $pipArguments += @('--index-url', 'https://pypi.org/simple') }
            # Avoid pip resolving unrelated, stale PATH devices to report where
            # entry-point scripts were installed. Do not edit the global PATH.
            $pipArguments += '--no-warn-script-location'
        }
        Invoke-AppPython $Exe $pipArguments
    } finally {
        foreach ($entry in @([Environment]::GetEnvironmentVariables('Process').GetEnumerator())) {
            if ($entry.Key -like 'PIP_*') { [Environment]::SetEnvironmentVariable($entry.Key, $null, 'Process') }
        }
        foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
    }
}

function Get-MosaicRootKey([string]$Root) {
    $hash = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes([IO.Path]::GetFullPath($Root).ToUpperInvariant())
        return ([BitConverter]::ToString($hash.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    } finally { $hash.Dispose() }
}

function Enter-MosaicBootstrapLock([string]$Path) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null
    try { return [IO.File]::Open($Path, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None) }
    catch { throw '同じアプリ環境のセットアップが実行中か、所有記録を開けません。先の操作が終わってから再試行してください。' }
}

function Read-MosaicBootstrapJournal([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $item.Length -gt 65536) {
        throw 'bootstrapの所有記録が通常のJSONファイルではありません。変更せず停止します。'
    }
    return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
}

function Write-MosaicBootstrapJournal([string]$Path, $Journal) {
    $Journal.updated_at = [DateTime]::UtcNow.ToString('o')
    $temporary = $Path + '.' + [guid]::NewGuid().ToString('N') + '.partial'
    $bytes = [Text.Encoding]::UTF8.GetBytes(($Journal | ConvertTo-Json -Depth 8))
    $stream = [IO.File]::Open($temporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) }
    finally { $stream.Dispose() }
    if (Test-Path -LiteralPath $Path) { [IO.File]::Replace($temporary, $Path, [NullString]::Value) }
    else { [IO.File]::Move($temporary, $Path) }
}

function Assert-MosaicJournalOwner($Journal, [string]$Kind, [string]$Root, [string]$BootstrapHome, [string]$Target) {
    if ($Journal.schema_version -ne 1 -or $Journal.kind -cne $Kind -or
        $Journal.owner_id -notmatch '^[a-f0-9]{32}$' -or
        [IO.Path]::GetFullPath([string]$Journal.app_home) -ine [IO.Path]::GetFullPath($BootstrapHome) -or
        [IO.Path]::GetFullPath([string]$Journal.target) -ine [IO.Path]::GetFullPath($Target) -or
        ($Root -and [IO.Path]::GetFullPath([string]$Journal.app_root) -ine [IO.Path]::GetFullPath($Root))) {
        throw 'bootstrapの所有記録と導入先が一致しません。既存ファイルを保全して停止します。'
    }
}

function Assert-MosaicPlainTarget([string]$Target) {
    if (Test-Path -LiteralPath $Target) {
        $item = Get-Item -LiteralPath $Target -Force
        if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw '未完成環境の導入先が通常のdirectoryではありません。上書きしません。'
        }
    }
}

function Assert-MosaicPythonArtifact($Spec) {
    $version = [string]$Spec.version
    if ($version -notmatch '^3\.13\.[0-9]+$' -or
        $Spec.url -cne ('https://www.python.org/ftp/python/' + $version + '/python-' + $version + '-amd64.exe') -or
        $Spec.sha256 -cnotmatch '^[a-f0-9]{64}$' -or
        $Spec.size_bytes -le 0 -or $Spec.size_bytes -gt 100MB -or
        $Spec.digest_source -cne ($Spec.url + '.sigstore')) {
        throw '保存された公式Python導入計画が不正です。別の取得元へ進みません。'
    }
}

function Initialize-MosaicPython([string]$Root, [string]$BootstrapHome, [string]$Target) {
    $python = Join-Path $Target 'python.exe'
    $journalPath = Join-Path $BootstrapHome 'python-bootstrap-journal.json'
    $lock = Enter-MosaicBootstrapLock ($journalPath + '.lock')
    try {
        $journal = Read-MosaicBootstrapJournal $journalPath
        if ($null -ne $journal) {
            Assert-MosaicJournalOwner $journal 'python' '' $BootstrapHome $Target
            if ($journal.phase -cnotin @('planned', 'installing', 'ready')) { throw 'Pythonの導入段階が不明です。記録を保全します。' }
            Assert-MosaicPythonArtifact $journal.artifact
        }
        # Reuse a complete compatible interpreter. Pending owned installers are
        # re-run instead, including when python.exe was created before a crash.
        if ((Test-Path -LiteralPath $python -PathType Leaf) -and ($null -eq $journal -or $journal.phase -eq 'ready')) {
            Invoke-AppPython $python @('-B', '-c', 'import sys,tkinter,venv,ensurepip;assert sys.version_info[:2]==(3,13)')
            return $python
        }
        Assert-MosaicPlainTarget $Target
        if ($null -eq $journal -and (Test-Path -LiteralPath $Target)) {
            throw 'Python導入先に所有を確認できない未完成ファイルがあります。上書きしません。'
        }
        if ($null -ne $journal -and $journal.phase -ne 'ready') {
            Assert-MosaicJournalOwner $journal 'python' $Root $BootstrapHome $Target
            $spec = $journal.artifact
        } else {
            # A completed receipt is not a permanent version pin when a later
            # explicit operation really needs a new interpreter download.
            $spec = Resolve-MosaicPython
            Assert-MosaicPythonArtifact $spec
            $journal = [pscustomobject]@{ schema_version = 1; kind = 'python'; owner_id = [guid]::NewGuid().ToString('N');
                app_root = [IO.Path]::GetFullPath($Root); app_home = [IO.Path]::GetFullPath($BootstrapHome);
                target = [IO.Path]::GetFullPath($Target); phase = 'planned'; artifact = $spec; updated_at = '' }
            Write-MosaicBootstrapJournal $journalPath $journal
        }
        $installer = Get-AppInstaller $spec ('python-' + $spec.version + '-x64.exe') 'Python Software Foundation'
        $journal.phase = 'installing'
        Write-MosaicBootstrapJournal $journalPath $journal
        if (-not (Test-Path -LiteralPath $Target)) { New-Item -ItemType Directory -Path $Target | Out-Null }
        $arguments = @('/quiet', 'InstallAllUsers=0', ('TargetDir="' + $Target + '"'),
            'Include_pip=1', 'Include_tcltk=1', 'Include_dev=1', 'Include_launcher=0',
            'Include_test=0', 'Include_doc=0', 'PrependPath=0', 'AssociateFiles=0', 'Shortcuts=0')
        $process = Start-Process -FilePath $installer -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
        if ($process.ExitCode -ne 0) { throw "Python導入に失敗しました（終了値$($process.ExitCode)）。所有記録と途中データを保持します。再実行で同じinstallerを使います。" }
        Invoke-AppPython $python @('-B', '-c', 'import sys,tkinter,venv,ensurepip;assert sys.version_info[:2]==(3,13)')
        $journal.phase = 'ready'
        Write-MosaicBootstrapJournal $journalPath $journal
        return $python
    } finally { $lock.Dispose() }
}

function Assert-MosaicAppVenv([string]$Python, [string]$Target) {
    Invoke-AppPython $Python @('-B', '-c',
        'import sys,tkinter;from pathlib import Path;assert (3,11)<=sys.version_info[:2]<(3,14);assert sys.prefix!=sys.base_prefix;assert Path(sys.prefix).resolve()==Path(sys.argv[1]).resolve()', $Target)
}

function Initialize-MosaicAppVenv([string]$Root, [string]$BootstrapHome) {
    $target = Join-Path $Root '.venv'
    $python = Join-Path $target 'Scripts/python.exe'
    $folder = Join-Path (Join-Path $BootstrapHome 'bootstrap-state') (Get-MosaicRootKey $Root)
    $journalPath = Join-Path $folder 'app-venv.json'
    $lock = Enter-MosaicBootstrapLock ($journalPath + '.lock')
    try {
        $journal = Read-MosaicBootstrapJournal $journalPath
        if ($null -ne $journal) {
            Assert-MosaicJournalOwner $journal 'app-venv' $Root $BootstrapHome $target
            if ($journal.phase -cnotin @('creating', 'ready')) { throw 'アプリvenvの導入段階が不明です。記録を保全します。' }
        }
        if ((Test-Path -LiteralPath $python -PathType Leaf) -and ($null -eq $journal -or $journal.phase -eq 'ready')) {
            Assert-MosaicAppVenv $python $target
            return $python
        }
        Assert-MosaicPlainTarget $target
        if ($null -eq $journal -and (Test-Path -LiteralPath $target)) {
            throw '所有を確認できない未完成のアプリ用.venvがあります。上書きしません。'
        }
        $base = Initialize-MosaicPython $Root $BootstrapHome (Join-Path $BootstrapHome 'python313')
        $baseHash = Get-MosaicFileSha256 $base
        if ($null -ne $journal -and ($journal.phase -ne 'ready' -or (Test-Path -LiteralPath $target))) {
            if ($journal.base_python -ine [IO.Path]::GetFullPath($base) -or $journal.base_python_sha256 -cne $baseHash) {
                throw '未完成venvを作成したPythonのpath/hashが変わりました。途中データを保全して停止します。'
            }
        } else {
            $journal = [pscustomobject]@{ schema_version = 1; kind = 'app-venv'; owner_id = [guid]::NewGuid().ToString('N');
                app_root = [IO.Path]::GetFullPath($Root); app_home = [IO.Path]::GetFullPath($BootstrapHome);
                target = [IO.Path]::GetFullPath($target); base_python = [IO.Path]::GetFullPath($base);
                base_python_sha256 = $baseHash; phase = 'creating'; updated_at = '' }
        }
        $journal.phase = 'creating'
        Write-MosaicBootstrapJournal $journalPath $journal
        # venv can finish its own partial directory in place; --clear and
        # recursive deletion are deliberately unnecessary.
        Invoke-AppPython $base @('-B', '-m', 'venv', $target)
        Assert-MosaicAppVenv $python $target
        $journal.phase = 'ready'
        Write-MosaicBootstrapJournal $journalPath $journal
        return $python
    } finally { $lock.Dispose() }
}
