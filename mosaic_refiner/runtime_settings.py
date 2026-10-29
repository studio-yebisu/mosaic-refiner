"""Explicit selection of one local ComfyUI. Import does not create an environment."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def settings_path():
    override = os.environ.get('MOSAIC_REFINER_RUNTIME_CONFIG')
    return Path(override) if override else Path(os.environ.get('LOCALAPPDATA', Path.home()/'.config'))/'MosaicRefiner/runtime.json'


def suggested_root():
    # Uses the current user's LOCALAPPDATA on Windows. This is a suggestion,
    # never a reason to create/move a second ComfyUI installation.
    return Path(os.environ.get('LOCALAPPDATA', Path.home()/'.local/share'))/'MosaicRefiner/ComfyUI'


@dataclass(frozen=True)
class RuntimeConnection:
    root: Path
    python: Path
    endpoint: str = 'http://127.0.0.1:8190'
    configured: bool = False

    def __post_init__(self):
        parsed = urlsplit(self.endpoint)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', '::1')
            or parsed.username or parsed.password or parsed.path not in ('', '/')
            or parsed.query or parsed.fragment or not parsed.port):
            raise ValueError('ComfyUI APIはhttpのloopbackアドレスと明示ポートを指定してください')
        object.__setattr__(self, 'endpoint', self.endpoint.rstrip('/'))
        object.__setattr__(self, 'root', Path(self.root).resolve())
        object.__setattr__(self, 'python', Path(self.python).resolve())

    @property
    def port(self):
        return urlsplit(self.endpoint).port

    def validate(self):
        if not self.configured:
            raise ValueError('初回セットアップで既存ComfyUIの場所・Python・APIを選択してください')
        missing = [] if (self.root/'main.py').is_file() else [str(self.root/'main.py')]
        missing.extend(str(self.root/name) for name in ('custom_nodes', 'models') if not (self.root/name).is_dir())
        if not self.python.is_file():
            missing.append(str(self.python))
        if missing:
            raise ValueError('選択した推論環境が見つかりません: '+', '.join(missing))
        return self

    def to_dict(self):
        return {'version': 1, 'root': str(self.root), 'python': str(self.python), 'endpoint': self.endpoint}


def load(path=None):
    path = Path(path) if path else settings_path()
    if not path.exists():
        root = suggested_root()
        return RuntimeConnection(root, root/'.venv/Scripts/python.exe')
    data = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(data, dict) or set(data) != {'version', 'root', 'python', 'endpoint'}
        or type(data['version']) is not int or data['version'] != 1
        or any(not isinstance(data[key], str) or not data[key].strip() for key in ('root','python','endpoint'))):
        raise ValueError('推論環境設定の形式が不正です。セットアップで選択し直してください')
    return RuntimeConnection(data['root'], data['python'], data['endpoint'], True)


def save(connection, path=None):
    connection.validate()
    from .checkpoint import atomic_json
    path = Path(path) if path else settings_path()
    atomic_json(path, connection.to_dict())
    return path


# Each application process binds to one selection. Changing it requires restart,
# so a running job cannot suddenly switch roots/Python/API.
SETTINGS_ERROR = None
try:
    RUNTIME = load()
except (ValueError, OSError) as exc:
    # Keep setup/help available so a malformed selection can be repaired.
    SETTINGS_ERROR = str(exc)
    _suggested = suggested_root()
    RUNTIME = RuntimeConnection(_suggested, _suggested/'.venv/Scripts/python.exe')


def selection_ready():
    try:
        RUNTIME.validate()
        return True
    except (ValueError, OSError):
        return False


def local_tool_bin():
    """Select app-owned versioned tools. A receipt cannot point outside this tree."""
    import re
    from hashlib import sha256
    tools=Path(os.environ.get('LOCALAPPDATA',Path.home()/'.cache'))/'MosaicRefiner/tools'
    pointer=tools/'ffmpeg-active.json'
    if pointer.is_file():
        try:
            digest=json.loads(pointer.read_text(encoding='utf-8'))['sha256']
            if not isinstance(digest,str) or not re.fullmatch('[a-f0-9]{64}',digest):return None
            directory=(tools/'ffmpeg-versions'/digest).resolve()
            if (tools/'ffmpeg-versions').resolve() not in directory.parents:return None
            record=json.loads((directory/'.mosaic-tools.json').read_text(encoding='utf-8'))
            if record.get('archive_sha256')!=digest:return None
            for name in ('ffmpeg.exe','ffprobe.exe'):
                executable=(directory/'bin'/name).resolve()
                if directory not in executable.parents:return None
                if sha256(executable.read_bytes()).hexdigest()!=record['executables'].get('bin/'+name):return None
            return directory/'bin'
        except (OSError,ValueError,KeyError,TypeError):return None
    # Preserve an older app-owned installation; no download or update at import.
    directory=tools/'ffmpeg'
    if (directory/'.mosaic-tools.json').is_file() and all((directory/'bin'/n).is_file() for n in ('ffmpeg.exe','ffprobe.exe')):
        return directory/'bin'
    return None


def use_local_tools():
    directory=local_tool_bin()
    if directory:
        os.environ['PATH']=str(directory)+os.pathsep+os.environ.get('PATH','')


use_local_tools()
