"""Explicit, resumable model acquisition. Importing this module performs no I/O."""
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .checkpoint import atomic_json, job_lock
from .model_auth import AuthError, Credentials, read_json, official_url, secure_open


class SetupError(ValueError):
    pass


@dataclass(frozen=True)
class Model:
    id: str
    repo: str
    revision: str
    filename: str
    target: str
    size_bytes: int
    sha256: str
    license_label: str
    terms_url: str
    source_checked_at: str
    provider: str = "huggingface"
    model_id: int | None = None
    version_id: int | None = None
    file_id: int | None = None

    @property
    def url(self):
        if self.provider == "civitai":
            return f"https://civitai.com/api/v1/model-versions/{self.version_id}"
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{quote(self.filename, safe='/')}"

    def destination(self, models_dir):
        root = Path(models_dir).resolve()
        relative = PurePosixPath(self.target)
        if relative.is_absolute() or any(x in ('..', '.') for x in relative.parts) or '\\' in self.target:
            raise SetupError("モデル配置先が不正です")
        target = (root / Path(*relative.parts)).resolve()
        if root not in target.parents:
            raise SetupError("モデル配置先がmodelsディレクトリの外です")
        return target


def catalog(path=None):
    data = json.loads((Path(path) if path else Path(__file__).with_name("model_catalog.json")).read_text(encoding="utf-8"))
    if not isinstance(data,dict) or data.get("schema_version") != 1:
        raise SetupError("モデル台帳の版が不明です")
    if not isinstance(data.get("models"),list) or not data["models"]:
        raise SetupError("モデル台帳には空でないmodels配列が必要です")
    try:
        models = [Model(**({"repo": "", "revision": ""} | m) if m.get("provider") == "civitai" else m)
                  for m in data["models"] if isinstance(m,dict)]
        if len(models)!=len(data["models"]):raise TypeError()
    except TypeError:
        raise SetupError("モデル台帳の項目が不足または不正です") from None
    if any(not isinstance(getattr(m,key),str) for m in models for key in (
            'id','repo','revision','filename','target','sha256','license_label','terms_url','source_checked_at','provider')):
        raise SetupError("モデル台帳の文字列項目の型が不正です")
    if len({m.id for m in models}) != len(models):
        raise SetupError("モデルIDが重複しています")
    for m in models:
        if (not re.fullmatch(r'[a-z0-9-]+', m.id)
                or not re.fullmatch(r'[0-9a-f]{64}', m.sha256)
                or type(m.size_bytes) is not int or m.size_bytes <= 0):
            raise SetupError("モデル台帳の固定版/hash/容量が不正です")
        if m.provider == "huggingface":
            if (not re.fullmatch(r'[0-9a-f]{40}', m.revision)
                    or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', m.repo)
                    or any(p in ("..", ".") for p in PurePosixPath(m.filename).parts)
                    or m.filename.startswith("/") or "\\" in m.filename or not m.filename):
                raise SetupError("Hugging Faceのrepo / commit / filenameが不正です")
        elif m.provider == "civitai":
            if any(type(v) is not int or v <= 0 for v in (m.model_id, m.version_id, m.file_id)):
                raise SetupError("Civitaiのmodel / version / file IDを固定してください")
        else:
            raise SetupError("モデル台帳の配布元に対応していません")
        m.destination(Path.cwd() / "models")
    return models


def download_url(model, credentials, opener=None):
    if model.provider != "civitai":
        return model.url
    info = read_json("civitai", model.url, credentials, opener)
    if not isinstance(info, dict) or info.get("id") != model.version_id or info.get("modelId") != model.model_id:
        raise SetupError("Civitaiの固定モデル版が一致しません")
    files=info.get("files")
    if not isinstance(files,list):
        raise SetupError("Civitaiのファイル情報を確認できません")
    matches = [f for f in files if isinstance(f,dict) and f.get("id") == model.file_id]
    if len(matches) != 1:
        raise SetupError("Civitaiの固定file IDがありません。別ファイルへ代用しません")
    selected = matches[0]
    hashes=selected.get("hashes")
    if (selected.get("name") != model.filename or not isinstance(hashes,dict)
            or not isinstance(hashes.get("SHA256"),str)
            or hashes["SHA256"].lower() != model.sha256):
        raise SetupError("Civitaiのfilename / SHA-256が台帳と一致しません")
    url = selected.get("downloadUrl", "")
    if not isinstance(url,str):
        raise SetupError("Civitaiの取得URLが不正です")
    from urllib.parse import urlsplit, parse_qsl
    parsed = urlsplit(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    keys = [key.lower() for key, _ in query]
    if (not official_url("civitai", url)
            or parsed.path != f"/api/download/models/{model.version_id}"
            or parsed.fragment or len(set(keys)) != len(keys) or keys.count('fileid') != 1
            or any(k.lower() not in ("type", "format", "size", "fp", "fileid") for k, _ in query)
            or any(k.lower() == 'fileid' and (k != 'fileId' or v != str(model.file_id)) for k, v in query)):
        raise SetupError("Civitaiの公式取得URLを確認できません")
    return url


def default_cache():
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "MosaicRefiner/model-downloads"


def checked_hash(path, cancelled=lambda: False):
    digest = sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            if cancelled():
                raise KeyboardInterrupt
            digest.update(block)
    return digest.hexdigest()


def inspect(models_dir, models=None, verify=False, cancelled=lambda: False, progress=lambda **kw: None):
    rows = []
    for m in models if models is not None else catalog():
        if cancelled():
            raise KeyboardInterrupt
        target = m.destination(models_dir)
        progress(phase="既存モデルを確認", model=m.id)
        state = "missing"
        if target.exists():
            state = "conflict"
            if target.is_file() and target.stat().st_size == m.size_bytes:
                state = "size-matches-unverified"
                if verify:
                    state = "verified" if checked_hash(target, cancelled) == m.sha256 else "conflict"
        rows.append({"id": m.id, "state": state, "size_bytes": m.size_bytes,
                     "target": str(target), "source_url": m.url, "terms_url": m.terms_url,
                     "license_label": m.license_label})
    return rows


def _space(path, needed):
    path = Path(path)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < needed + 64 * 1024 * 1024:
        raise SetupError("ダウンロード/配置用の空き容量が不足しています")


def download(m, cache_dir, cancelled=lambda: False, progress=lambda **kw: None, opener=None, credentials=None):
    """HTTP Range + pinned size/hash; interrupted bytes remain exclusively in cache."""
    if cancelled():raise KeyboardInterrupt
    if m.provider == 'civitai':
        from .lora_setup import authenticated_credentials
        try:
            credentials = authenticated_credentials(credentials, opener)
        except AuthError as exc:
            raise SetupError(str(exc)) from None
    directory = Path(cache_dir) / m.id
    directory.mkdir(parents=True, exist_ok=True)
    with job_lock(directory):
        identity = {"url": m.url, "size_bytes": m.size_bytes, "sha256": m.sha256}
        if m.provider=="civitai":
            identity.update(provider=m.provider,model_id=m.model_id,version_id=m.version_id,file_id=m.file_id)
        receipt = directory / "source.json"
        partial, complete = directory / "download.part", directory / "verified.safetensors"
        if receipt.exists():
            if json.loads(receipt.read_text(encoding='utf-8')) != identity:
                raise SetupError("保存された取得元と台帳が異なります。別cacheを指定してください")
        else:
            if partial.exists() or complete.exists():
                raise SetupError("取得元を確認できないcacheがあります。別cacheを指定してください")
            atomic_json(receipt, identity)
        if complete.exists():
            if complete.stat().st_size == m.size_bytes and checked_hash(complete, cancelled) == m.sha256:
                return complete
            raise SetupError("検証済みcacheが変更されています。別cacheを指定してください")
        credentials = credentials if credentials is not None else Credentials.saved()
        try:
            url = download_url(m, credentials, opener)
        except AuthError as exc:
            raise SetupError(str(exc)) from None
        failures = 0
        while True:
            if cancelled():
                raise KeyboardInterrupt
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > m.size_bytes:
                raise SetupError("途中ファイルの容量が台帳より大きいため停止しました")
            if offset == m.size_bytes:
                break
            _space(directory, m.size_bytes - offset)
            headers = {"Range": f"bytes={offset}-", "Accept-Encoding": "identity",
                       "User-Agent": "mosaic-refiner-model-setup/1"}
            try:
                headers.update(credentials.headers(m.provider, url))
            except AuthError as exc:
                raise SetupError(str(exc)) from None
            try:
                with (opener or secure_open)(Request(url, headers=headers), timeout=30) as response:
                    status = response.status
                    content_type = response.headers.get('Content-Type', '').split(';',1)[0].strip().lower()
                    if content_type.startswith('text/') or content_type == 'application/json' or content_type.endswith('+json'):
                        raise SetupError('配布元がモデル以外の応答を返しました。本人の取得権限・同意・配布状態を確認してください')
                    if status == 206:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                        if (not match or int(match[1]) != offset or int(match[3]) != m.size_bytes
                                or not offset <= int(match[2]) < m.size_bytes):
                            raise SetupError("配布元の再開位置/容量が一致しません")
                        response_end = int(match[2]) + 1
                    elif status == 200:
                        response_end = m.size_bytes
                    else:
                        raise SetupError(f"配布元の応答が不正です（HTTP {status}）")
                    length = response.headers.get('Content-Length')
                    if length is not None:
                        try: declared = int(length)
                        except (ValueError,TypeError): raise SetupError('配布元の応答容量が不正です') from None
                        expected = response_end - (offset if status == 206 else 0)
                        if declared != expected:
                            raise SetupError('配布元の応答容量が固定モデルと一致しません。途中データは変更しません')
                    if status == 200:
                        # Some providers ignore Range. Restart only our own partial.
                        offset = 0
                        _space(directory, m.size_bytes)
                    with partial.open('ab' if status == 206 else 'wb') as stream:
                        written = offset
                        while True:
                            if cancelled():
                                raise KeyboardInterrupt
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            if written + len(block) > response_end:
                                raise SetupError("配布元のファイルが台帳の容量を超えています")
                            stream.write(block)
                            written += len(block)
                            progress(phase="ダウンロード", model=m.id, downloaded=written, total=m.size_bytes)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if written != m.size_bytes:
                        raise OSError("incomplete response")
            except HTTPError as exc:
                if exc.code in (401, 403):
                    raise SetupError("配布元が取得を拒否しました。配布ページで利用条件・アクセス権を確認してください") from None
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise SetupError(f"配布元から取得できません（HTTP {exc.code}）") from None
                failures += 1
            except AuthError as exc:
                raise SetupError(str(exc)) from None
            except (OSError, URLError):
                failures += 1
            else:
                break
            if failures >= 3:
                raise SetupError("通信が中断しました。途中データは保存済みです。再開してください")
            progress(phase="通信待ち・再試行", model=m.id)
            for _ in range(10 * failures):
                if cancelled():
                    raise KeyboardInterrupt
                time.sleep(.1)
        progress(phase="SHA-256を検証", model=m.id)
        if checked_hash(partial, cancelled) != m.sha256:
            partial.rename(directory / f"hash-mismatch-{time.time_ns()}.part")
            raise SetupError("取得したモデルのhashが一致しません。配置せずcacheへ保全しました")
        if cancelled():
            raise KeyboardInterrupt
        os.replace(partial, complete)
        return complete


def _record_terms_acknowledgement(model, cache_dir):
    from datetime import datetime, timezone
    receipt_dir=cache_dir/model.id
    receipt_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(receipt_dir/'terms-confirmed.json',{
        'id':model.id,'provider':model.provider,'source_url':model.url,'sha256':model.sha256,
        'terms_url':model.terms_url,'confirmed_at':datetime.now(timezone.utc).isoformat(),
        'acknowledgement_kind':'user_terms_review','model_bytes_verified':True,
        'license_grant_verified':False,'provider_approval_verified':False})


def install(models_dir, cache_dir=None, models=None, cancelled=lambda: False, progress=lambda **kw: None,
            credentials=None,terms_confirmed=False):
    models = catalog() if models is None else models
    cache_dir = default_cache() if cache_dir is None else Path(cache_dir)
    rows = inspect(models_dir, models, verify=True, cancelled=cancelled, progress=progress)
    if any(row['state'] == 'conflict' for row in rows):
        raise SetupError("同名の別モデルがあります。上書きせず停止しました。配置先を確認してください")
    for m, row in zip(models, rows):
        if cancelled():
            raise KeyboardInterrupt
        if row['state'] == 'verified':
            if terms_confirmed:_record_terms_acknowledgement(m,cache_dir)
            progress(phase="既存モデルを再利用", model=m.id)
            continue
        target = m.destination(models_dir)
        _space(target.parent, m.size_bytes)
        cached = download(m, cache_dir, cancelled, progress, credentials=credentials)
        # Cache and models may share a drive. Recheck after downloading consumes
        # space, before allocating the verified destination copy.
        _space(target.parent, m.size_bytes)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.mosaic-{time.time_ns()}.partial")
        try:
            progress(phase="配置", model=m.id)
            with cached.open('rb') as old, temporary.open('xb') as new:
                while block := old.read(4 * 1024 * 1024):
                    if cancelled():
                        raise KeyboardInterrupt
                    new.write(block)
                new.flush()
                os.fsync(new.fileno())
            if checked_hash(temporary, cancelled) != m.sha256:
                raise SetupError("配置前のhash検査に失敗しました")
            if cancelled():
                raise KeyboardInterrupt
            os.link(temporary, target)  # Atomic publication, never overwrite.
        finally:
            temporary.unlink(missing_ok=True)
        if terms_confirmed:_record_terms_acknowledgement(m,cache_dir)
    return {"state": "models-ready", "runtime_ready": False,
            "models": [{"id": m.id, "state": "verified", "target": str(m.destination(models_dir))} for m in models]}
