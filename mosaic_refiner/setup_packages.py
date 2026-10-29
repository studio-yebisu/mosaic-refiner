"""Explicit resolution of official Windows GPU wheels; no install or import side effects."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from hashlib import sha256
from html.parser import HTMLParser
import json
import re
from urllib.parse import unquote, urldefrag, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from .model_setup import SetupError


TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
SECURITY_FLOOR = Version("2.14.1")
AUDIO_ABI_SOURCE = "https://docs.pytorch.org/audio/stable/installation.html#compatibility-matrix"


def require_secure_torch(value):
    """Reject unfixed/unknown running versions without importing Torch."""
    try:
        version = Version(str(value))
    except InvalidVersion as exc:
        raise SetupError("実行中Torchの正式版を確認できません") from exc
    if (version.is_prerelease or version.is_devrelease
            or Version(version.base_version) < SECURITY_FLOOR):
        raise SetupError("Torchの既知不具合を含む構成です。公式修正を含む安定版2.14.1以上が必要です")
    if version.local != "cu130":
        raise SetupError("実行中TorchはCUDA 13.0用の公式安定版（+cu130）が必要です")
    return str(version)


def _official(url):
    try:
        p = urlsplit(url)
        if (p.scheme != "https" or p.username or p.password or p.port not in (None, 443)
                or p.query or p.fragment):
            return False
        if p.hostname in ("download.pytorch.org", "download-r2.pytorch.org"):
            return p.path.startswith("/whl/cu130/")
        if p.hostname == "pypi.org":
            return p.path.startswith("/simple/")
        if p.hostname == "files.pythonhosted.org":
            return p.path.startswith("/packages/")
    except (TypeError, ValueError):
        pass
    return False


class _Redirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _official(newurl):
            raise SetupError("package情報を公式HTTPS配布元以外へ転送しません")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _Reader:
    def __init__(self, opener):
        self.opener = opener or build_opener(_Redirect()).open
        self.cache = {}

    def read(self, url, *, metadata=False, accept=None):
        if not _official(url):
            raise SetupError("package情報の取得先が公式HTTPS配布元ではありません")
        if url not in self.cache:
            headers = {"User-Agent": "MosaicRefiner/setup"}
            if accept:
                headers["Accept"] = accept
            limit = 512 * 1024 if metadata else 8 * 1024 * 1024
            try:
                with self.opener(Request(url, headers=headers), timeout=30) as response:
                    if hasattr(response, "geturl") and not _official(response.geturl()):
                        raise SetupError("package情報の取得先が公式配布元から変わりました")
                    data = response.read(limit + 1)
                if len(data) > limit:
                    raise SetupError("package情報の取得容量が上限を超えています")
                self.cache[url] = data
            except SetupError:
                raise
            except (OSError, ValueError) as exc:
                raise SetupError("公式package情報を取得できません。接続を確認して再試行してください") from exc
        return self.cache[url]


def _digest(value):
    return value.lower() if isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) else None


@dataclass(frozen=True)
class _Wheel:
    name: str
    version: Version
    filename: str
    tags: frozenset
    url: str
    sha256: str | None
    metadata_sha256: str | None
    requires_python: str = ""

    def record(self):
        if not self.sha256:
            raise SetupError("選択packageの公開wheel SHA-256がありません。旧版へ代用しません")
        return {"version": str(self.version), "filename": self.filename, "url": self.url,
                "sha256": self.sha256, "metadata_url": self.url + ".metadata",
                "metadata_sha256": self.metadata_sha256}


class _Index(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs))


def _wheel(filename, url, archive_hash, metadata_hash, requires_python=""):
    try:
        name, version, _, tags = parse_wheel_filename(filename)
    except (ValueError, InvalidVersion):
        return None
    if version.is_prerelease or version.is_devrelease:
        return None
    if not _official(url):
        raise SetupError("wheelの取得先が公式HTTPS配布元ではありません")
    return _Wheel(canonicalize_name(name), version, filename, tags, url,
                  _digest(archive_hash), _digest(metadata_hash), requires_python or "")


def _torch_wheels(reader, package):
    url = TORCH_INDEX + "/" + package + "/"
    index = _Index()
    try:
        index.feed(reader.read(url).decode("utf-8"))
    except UnicodeError as exc:
        raise SetupError("公式Torch indexの文字形式が不正です") from exc
    result = []
    for entry in index.links:
        if "data-yanked" in entry:
            continue
        original = urljoin(url, entry.get("href", ""))
        address, fragment = urldefrag(original)
        filename = unquote(urlsplit(address).path.rsplit("/", 1)[-1])
        if not filename.endswith(".whl"):
            continue
        # The CUDA index also lists historical CPU/macOS wheels outside the
        # cu130 directory. Ignore those before enforcing the selected origin.
        try:
            name, version, _, _ = parse_wheel_filename(filename)
        except (ValueError, InvalidVersion):
            continue
        if canonicalize_name(name) != package or version.local != "cu130":
            continue
        archive_hash = fragment.removeprefix("sha256=") if fragment.startswith("sha256=") else None
        metadata_hash = entry.get("data-core-metadata", entry.get("data-dist-info-metadata", ""))
        metadata_hash = metadata_hash.removeprefix("sha256=")
        item = _wheel(filename, address, archive_hash, metadata_hash, entry.get("data-requires-python", ""))
        if item and item.name == package and item.version.local == "cu130":
            result.append(item)
    return result


def _pypi_wheels(reader, package):
    url = "https://pypi.org/simple/" + package + "/"
    try:
        document = json.loads(reader.read(url, accept="application/vnd.pypi.simple.v1+json"))
        entries = document["files"]
        if not isinstance(entries, list):
            raise ValueError()
    except (ValueError, KeyError, TypeError) as exc:
        raise SetupError("PyPIの公開package情報を確認できません") from exc
    result = []
    for entry in entries:
        if entry.get("yanked") or not entry.get("filename", "").endswith(".whl"):
            continue
        core = entry.get("core-metadata", entry.get("dist-info-metadata", {}))
        item = _wheel(entry["filename"], urljoin(url, entry["url"]),
                      entry.get("hashes", {}).get("sha256"),
                      core.get("sha256") if isinstance(core, dict) else None,
                      entry.get("requires-python", ""))
        if item and item.name == package:
            result.append(item)
    return result


def _python_ok(specifier, python):
    try:
        return not specifier or SpecifierSet(specifier).contains(python)
    except (InvalidSpecifier, TypeError) as exc:
        raise SetupError("packageのPython互換条件が不正です") from exc


def _metadata(reader, item):
    if not item.metadata_sha256:
        raise SetupError("PEP 658 metadataの公開SHA-256がありません。旧版へ代用しません")
    data = reader.read(item.url + ".metadata", metadata=True)
    if sha256(data).hexdigest() != item.metadata_sha256:
        raise SetupError("wheel metadataのSHA-256が公開値と一致しません")
    message = BytesParser(policy=default).parsebytes(data)
    try:
        if (len(message.get_all("Name", [])) != 1 or len(message.get_all("Version", [])) != 1
                or canonicalize_name(message["Name"]) != item.name
                or Version(message["Version"]) != item.version):
            raise ValueError()
        requirements = [Requirement(value) for value in message.get_all("Requires-Dist", [])]
    except (InvalidVersion, InvalidRequirement, ValueError, TypeError) as exc:
        raise SetupError("wheel metadataのpackage名・版・依存条件が一致しません") from exc
    return message.get("Requires-Python", ""), requirements


def _ordered(items, ranks):
    matches = [(item, min((ranks[t] for t in item.tags if t in ranks), default=None)) for item in items]
    return [item for item, _ in sorted((row for row in matches if row[1] is not None),
                                      key=lambda row: (row[0].version, -row[1], row[0].filename), reverse=True)]


def _environment(python, linux=False):
    values = default_environment()
    values.update(python_version=".".join(str(x) for x in python.release[:2]),
                  python_full_version=str(python), implementation_name="cpython",
                  platform_python_implementation="CPython", platform_machine="x86_64",
                  platform_system="Linux" if linux else "Windows",
                  sys_platform="linux" if linux else "win32", os_name="posix" if linux else "nt",
                  extra="")
    return values


def _torch_requirements(requirements, python):
    return [r for r in requirements if canonicalize_name(r.name) == "torch"
            and (not r.marker or r.marker.evaluate(_environment(python)))]


def _companion(reader, items, ranks, python, torch_version, *, audio=False):
    for item in _ordered(items, ranks):
        if not _python_ok(item.requires_python, python):
            continue
        python_spec, requirements = _metadata(reader, item)
        if not _python_ok(python_spec, python):
            continue
        constraints = _torch_requirements(requirements, python)
        if constraints and all(r.specifier and torch_version in r.specifier for r in constraints):
            return item
        if audio and not constraints:
            # TorchAudio 2.11 has a documented stable ABI for PyTorch >=2.11.
            # Do not infer the ABI baseline of a future, unannotated minor.
            if item.version.release[:2] == (2, 11) and torch_version >= Version("2.11"):
                return item
            if item.version >= Version("2.11"):
                raise SetupError("TorchAudioの新しい版のABI条件が確認できません。旧版へ代用しません")
    raise SetupError("選択Torchと互換の公式Windows companion wheelがありません")


def _triton_requirement(reader, torch_items, selected, python):
    interpreter = "cp" + "".join(str(x) for x in python.release[:2])
    linux = [item for item in torch_items if item.version == selected.version and any(
        tag.interpreter == interpreter and tag.abi == interpreter
        and tag.platform.endswith("x86_64") and tag.platform.startswith("manylinux") for tag in item.tags)]
    if not linux:
        raise SetupError("同じTorch版の公式Triton依存metadataがありません")
    source = sorted(linux, key=lambda item: item.filename)[0]
    _, requirements = _metadata(reader, source)
    active = [r for r in requirements if canonicalize_name(r.name) == "triton"
              and (not r.marker or r.marker.evaluate(_environment(python, linux=True)))]
    minors = set()
    for requirement in active:
        for specifier in requirement.specifier:
            if specifier.operator in ("==", "~=", ">=") and "*" not in specifier.version:
                version = Version(specifier.version)
                if not version.is_prerelease and len(version.release) >= 2:
                    minors.add(version.release[:2])
    if len(minors) != 1:
        raise SetupError("Torchが要求するTriton major/minorを一意に確認できません")
    return next(iter(minors)), {"metadata_url": source.url + ".metadata",
                               "metadata_sha256": source.metadata_sha256,
                               "requires_dist": [str(r) for r in active]}


def _pypi_selected(reader, package, ranks, python, *, minor=None):
    for item in _ordered(_pypi_wheels(reader, package), ranks):
        if minor and item.version.release[:2] != minor:
            continue
        if not _python_ok(item.requires_python, python):
            continue
        python_spec, _ = _metadata(reader, item)
        if _python_ok(python_spec, python):
            return item
    raise SetupError("指定Pythonに互換の公式PyPI stable wheelがありません: " + package)


def _replace(requirements, updates):
    result, seen = [], set()
    for text in requirements:
        try:
            name = canonicalize_name(Requirement(text).name)
        except (InvalidRequirement, TypeError) as exc:
            raise SetupError("既存のpackage要求形式が不正です") from exc
        if name in updates:
            result.append(name + "==" + str(updates[name].version))
            seen.add(name)
        else:
            result.append(text)
    result.extend(name + "==" + str(item.version) for name, item in updates.items() if name not in seen)
    return result


def resolve_packages(recipe, python_version, opener=None):
    """Resolve once on explicit setup; return an independently owned pinned recipe.

    Only public index/JSON and hash-verified PEP 658 metadata are fetched. Wheel
    bodies, credentials, model files and the existing runtime are never touched.
    """
    if not isinstance(recipe, dict) or recipe.get("torch_index", "").rstrip("/") != TORCH_INDEX:
        raise SetupError("GPU packageの解決先は公式CUDA 13.0 indexへ限定します")
    try:
        text = ".".join(str(x) for x in python_version) if isinstance(python_version, (list, tuple)) else str(python_version)
        if not re.fullmatch(r"3\.\d+(?:\.\d+)?", text):
            raise ValueError()
        python = Version(text)
        major, minor = python.release[:2]
    except (ValueError, InvalidVersion) as exc:
        raise SetupError("Python版は3.11.15のような正式版を指定してください") from exc
    reader = _Reader(opener)
    target_tags = list(cpython_tags((major, minor), abis=[f"cp{major}{minor}"], platforms=["win_amd64"]))
    target_tags += list(compatible_tags((major, minor), interpreter=f"cp{major}{minor}", platforms=["win_amd64"]))
    ranks = {tag: index for index, tag in reversed(list(enumerate(target_tags)))}
    torch_items = _torch_wheels(reader, "torch")
    candidates = _ordered(torch_items, ranks)
    if not candidates:
        raise SetupError("指定Python/Windows x64用の公式CUDA 13.0 stable Torchがありません")
    torch = candidates[0]
    require_secure_torch(torch.version)
    python_spec, _ = _metadata(reader, torch)
    if not _python_ok(torch.requires_python, python) or not _python_ok(python_spec, python):
        raise SetupError("最新Torch wheelのPython条件を満たしません。旧版へ代用しません")
    vision = _companion(reader, _torch_wheels(reader, "torchvision"), ranks, python, torch.version)
    audio = _companion(reader, _torch_wheels(reader, "torchaudio"), ranks, python, torch.version, audio=True)
    triton_minor, triton_source = _triton_requirement(reader, torch_items, torch, python)
    triton = _pypi_selected(reader, "triton-windows", ranks, python, minor=triton_minor)
    pip = _pypi_selected(reader, "pip", ranks, python)
    selected = {item.name: item for item in (torch, vision, audio, triton, pip)}
    records = {name: item.record() for name, item in selected.items()}
    result = deepcopy(recipe)
    result["gpu_requirements"] = _replace(result.get("gpu_requirements", []),
                                          {item.name: item for item in (torch, vision, audio)})
    result["requirements"] = _replace(result.get("requirements", []), {triton.name: triton})
    result["bootstrap_pip_version"] = str(pip.version)
    resolution = deepcopy(result.get("resolution", {}))
    resolution.update(resolved_at=datetime.now(timezone.utc).isoformat(), python_version=str(python),
                      torch_security_floor=str(SECURITY_FLOOR), packages=records,
                      torch_triton_source=triton_source, torchaudio_abi_source=AUDIO_ABI_SOURCE)
    result["resolution"] = resolution
    return result
