"""Pinned downloads and traversal-safe, collision-preserving publication."""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

from .checkpoint import atomic_json
from .model_auth import Credentials
from .model_setup import SetupError, download, checked_hash, default_cache


@dataclass(frozen=True)
class Artifact:
    id: str
    url: str
    size_bytes: int
    sha256: str
    provider: str = field(default="public-artifact", init=False)

    @classmethod
    def from_dict(cls, data):
        result=cls(**{k:data[k] for k in ('id','url','size_bytes','sha256')})
        if (not result.url.startswith('https://') or not re.fullmatch('[a-z0-9-]+',result.id)
            or not re.fullmatch('[a-f0-9]{64}',result.sha256) or result.size_bytes<=0):
            raise SetupError('固定取得元・容量・SHA-256が不正です')
        return result


def acquire(data, cancelled=lambda:False, progress=lambda **kw:None):
    # Code/tool archives are public pinned assets, separate from model access.
    # Do not read the user's model credentials for an unauthenticated download.
    artifact=Artifact.from_dict(data)
    if data.get('cache_path'):
        cache_root=(default_cache().parent/'setup-resolution').resolve()
        cached=Path(data['cache_path']).resolve()
        if cache_root not in cached.parents:
            raise SetupError('解決済みコードcacheが専用cacheの外です')
        if cached.exists():
            if not cached.is_file() or cached.stat().st_size!=artifact.size_bytes or checked_hash(cached,cancelled)!=artifact.sha256:
                raise SetupError('解決済みコードcacheの容量/hashが変更されています')
            return cached
    return download(artifact, default_cache().parent/'setup-downloads', cancelled, progress,
                    credentials=Credentials())


def tree_hashes(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


def unpack(archive, directory, strip_root=True, cancelled=lambda:False):
    """Validate all members before writing any. No symlinks, drives, or escapes."""
    directory=Path(directory).resolve()
    with zipfile.ZipFile(archive) as z:
        members=z.infolist()
        if sum(m.file_size for m in members)>1024**3:
            raise SetupError('コード/tool archiveの展開容量が上限を超えています')
        roots=set()
        plan=[]
        for m in members:
            p=PurePosixPath(m.filename)
            if (p.is_absolute() or '\\' in m.orig_filename or ':' in m.filename
                or any(s in ('..','.') for s in p.parts) or stat.S_ISLNK(m.external_attr>>16)):
                raise SetupError('archiveに安全でない配置先があります')
            if any(s.endswith((' ','.')) or s.split('.')[0].upper() in
                   {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}
                   for s in p.parts):
                raise SetupError('archiveにWindowsで使えない配置先があります')
            if not p.parts: continue
            roots.add(p.parts[0])
            parts=p.parts[1:] if strip_root else p.parts
            if not parts: continue
            target=directory.joinpath(*parts).resolve()
            if directory not in target.parents: raise SetupError('archiveの配置先がrootの外です')
            plan.append((m,target))
        if strip_root and len(roots)!=1: raise SetupError('archiveのrootが一つではありません')
        if len({str(p).casefold() for m,p in plan})!=len(plan):
            raise SetupError('archiveに同名の配置先があります')
        for m,target in plan:
            if cancelled(): raise KeyboardInterrupt
            if m.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(m) as old, target.open('xb') as new: shutil.copyfileobj(old,new)


def publish_archive(data, target, cancelled=lambda:False, progress=lambda **kw:None):
    target=Path(target).resolve()
    if target.exists(): raise SetupError('既存ディレクトリへコードを上書きしません: '+str(target))
    archive=acquire(data,cancelled,progress)
    target.parent.mkdir(parents=True,exist_ok=True)
    # Only this newly created staging directory may be cleaned on failure.
    with tempfile.TemporaryDirectory(prefix='.mosaic-setup-',dir=target.parent) as stage:
        stage=Path(stage).resolve()
        if target.parent not in stage.parents: raise SetupError('stagingの場所が不正です')
        payload=stage/'payload'; payload.mkdir()
        unpack(archive,payload,cancelled=cancelled)
        hashes=tree_hashes(payload)
        # Publish the ownership record in the same rename as the code. A crash
        # after rename can be recovered without adopting an unknown directory.
        atomic_json(payload/'.mosaic-source.json', {'archive_sha256':data['sha256'],
                    'url':data['url'],'files':hashes})
        if cancelled(): raise KeyboardInterrupt
        if target.exists(): raise SetupError('取得中に配置先が作成されました。上書きしません')
        # os.rename on Windows refuses an existing target.
        os.rename(payload,target)
    return {'archive_sha256':data['sha256'],'files':hashes}


def verify_published(data, target, allowed=None):
    target=Path(target)
    marker=target/'.mosaic-source.json'
    if not marker.is_file():raise SetupError('導入先の所有記録がありません。上書きしません')
    record=json.loads(marker.read_text(encoding='utf-8'))
    if record.get('archive_sha256')!=data['sha256'] or record.get('url')!=data['url']:
        raise SetupError('導入先の取得元/版が異なります。上書きしません')
    for name,digest in record['files'].items():
        path=(target/name).resolve()
        if target.resolve() not in path.parents:raise SetupError('導入記録の配置先が不正です')
        actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual!=digest and actual not in (allowed or {}).get(name,[]):
            raise SetupError('導入後の変更があります。上書きしません: '+name)
    return {'archive_sha256':data['sha256'],'files':record['files'],'state':'verified'}


def verify_existing_source(data,target,cancelled=lambda:False,progress=lambda **kw:None):
    """Compare executable/text source, allowing Windows Git CRLF only."""
    archive=acquire(data,cancelled,progress)
    checked=[]
    with zipfile.ZipFile(archive) as z:
        prefix=z.namelist()[0].split('/')[0]+'/'
        for m in z.infolist():
            if cancelled():raise KeyboardInterrupt
            relative=m.filename.removeprefix(prefix)
            if m.is_dir() or not relative.endswith(('.py','.toml','.txt')): continue
            if relative.startswith(('tests/','test/')): continue
            path=Path(target)/relative
            if not path.is_file() or path.read_bytes().replace(b'\r\n',b'\n')!=z.read(m).replace(b'\r\n',b'\n'):
                raise SetupError('必要ノードの版/利用者の変更が異なります。保全して停止します: '+str(path))
            checked.append(relative)
    return {'state':'verified','source_files':len(checked),'revision':data['revision']}
