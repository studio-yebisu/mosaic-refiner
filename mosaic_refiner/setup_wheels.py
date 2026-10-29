"""Explicit wheel-only dependency lock for one newly owned ComfyUI venv.

The caller owns the setup lock and isolates pip's environment/configuration in
run_command. No subprocess or network I/O occurs on import. A pip report is an
installation plan, not an outbound network firewall: all of its selected wheel
origins are checked before installation.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlsplit

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from .checkpoint import atomic_json
from .model_setup import SetupError
from .setup_packages import require_secure_torch, TORCH_INDEX


PYPI_INDEX = "https://pypi.org/simple"
LOCK_NAME = "dependency-wheel-lock.json"
DIRECT = ("torch", "torchvision", "torchaudio", "triton-windows")
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_PACKAGES = 512
PROBE_CODE = """import importlib.metadata as m,json,platform,struct,sys,sysconfig
data={'python_version':list(sys.version_info[:3]),'implementation':sys.implementation.name,
      'platform':sys.platform,'bits':struct.calcsize('P')*8,
      'free_threaded':bool(sysconfig.get_config_var('Py_GIL_DISABLED')),
      'installed':[{'name':d.metadata['Name'],'version':d.version}
                   for d in m.distributions() if d.metadata['Name']]}
with open(sys.argv[1],'w',encoding='utf-8') as f:json.dump(data,f)
"""


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value):
        raise SetupError("wheelの公開SHA-256が不足または不正です")
    return value.lower()


def _version(value):
    try:
        parsed = Version(value)
    except (InvalidVersion, TypeError) as exc:
        raise SetupError("依存packageの正式版を確認できません") from exc
    if parsed.is_prerelease or parsed.is_devrelease:
        raise SetupError("依存に未確定の開発版が含まれます。安定版へ勝手に代用しません")
    return str(parsed)


def _requirements(lines):
    if not isinstance(lines, list) or any(not isinstance(line, str) for line in lines):
        raise SetupError("依存要求は文字列の配列が必要です")
    result = []
    for raw in lines:
        line = raw.strip().split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = Requirement(line)
        except InvalidRequirement as exc:
            raise SetupError("依存要求にpackage以外の指示があります。VCS・local・index・別requirementsは許可しません") from exc
        if item.url is not None or any(c in raw for c in ("\r", "\n", "\x00")):
            raise SetupError("依存要求の外部URL・VCS・local・複数行指定は許可しません")
        result.append(item)
    return result


def _wheel(name, version, url, digest, tags):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise SetupError("wheelのpackage名が不正です")
    name, version = canonicalize_name(name), _version(version)
    try:
        p = urlsplit(url)
        path = unquote(p.path)
        valid = (p.scheme == "https" and not p.username and not p.password and p.port in (None, 443)
                 and not p.query and not p.fragment and not any(ord(c) <= 32 or ord(c) == 127 for c in url)
                 and "\\" not in path and not set(path.split("/")) & {".", ".."})
        valid = valid and ((p.hostname == "files.pythonhosted.org" and path.startswith("/packages/"))
                           or (p.hostname in ("download.pytorch.org", "download-r2.pytorch.org")
                               and path.startswith("/whl/cu130/")))
        if not valid:
            raise ValueError()
        filename = path.rsplit("/", 1)[-1]
        wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(filename)
        if canonicalize_name(wheel_name) != name or wheel_version != Version(version) or not set(wheel_tags) & tags:
            raise ValueError()
    except (ValueError, TypeError) as exc:
        raise SetupError("公式配布元の対象ABI向けwheel URL・名称・版が一致しません") from exc
    return {"name": name, "version": version, "url": url, "sha256": _hash(digest), "filename": filename}


def _read_json(path):
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError()
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (OSError, ValueError, TypeError) as exc:
        raise SetupError("依存の計画・検査記録が未完成または不正です。旧版へ再解決しません") from exc


def _temporary(folder, suffix):
    descriptor, name = tempfile.mkstemp(prefix="mosaic-dependencies-", suffix=suffix, dir=folder)
    os.close(descriptor)
    return Path(name)


def _remove(path):
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass  # A private, owned temporary file must not mask the primary result.


def _probe(connection, folder, run_command, cancelled, progress, log):
    path = _temporary(folder, ".probe.json")
    try:
        run_command([connection.python, "-B", "-c", PROBE_CODE, path], connection.root, log, cancelled, progress)
        record = _read_json(path)
    finally:
        _remove(path)
    python = record.get("python_version")
    if (not isinstance(python, list) or len(python) != 3 or any(type(v) is not int or v < 0 for v in python)
            or python[0] != 3 or record.get("implementation") != "cpython" or record.get("platform") != "win32"
            or record.get("bits") != 64 or record.get("free_threaded") is not False):
        raise SetupError("新規GPU環境のWindows x64 CPython ABIを確認できません")
    installed = record.get("installed")
    if not isinstance(installed, list):
        raise SetupError("導入済みpackageの一覧が不正です")
    versions = {}
    for item in installed:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise SetupError("導入済みpackageの一覧が不正です")
        name = canonicalize_name(item["name"])
        if name in versions:
            raise SetupError("同名の導入済みpackageが重複しています。環境を変更しません")
        versions[name] = _version(item.get("version"))
    return tuple(python), versions


def _environment(python):
    result = default_environment()
    result.update(python_version=".".join(map(str, python[:2])), python_full_version=".".join(map(str, python)),
                  implementation_version=".".join(map(str, python)),
                  implementation_name="cpython", platform_python_implementation="CPython", os_name="nt",
                  sys_platform="win32", platform_system="Windows", platform_machine="AMD64", extra="")
    return result


def _active(requirement, environment, extras=()):
    return not requirement.marker or any(requirement.marker.evaluate({**environment, "extra": extra}) for extra in ("", *extras))


def _selected(data, tags):
    raw = data.get("resolution", {}).get("packages", {})
    if not isinstance(raw, dict) or any(name not in raw for name in (*DIRECT, "pip")):
        raise SetupError("公式GPU wheelとpipの解決済み記録が不足しています")
    selected = {}
    for name in (*DIRECT, "pip"):
        row = raw[name]
        if not isinstance(row, dict):
            raise SetupError("公式GPU wheelの記録形式が不正です")
        item = _wheel(name, row.get("version"), row.get("url"), row.get("sha256"), tags)
        if name in ("torch", "torchvision", "torchaudio"):
            if Version(item["version"]).local != "cu130" or urlsplit(item["url"]).hostname not in ("download.pytorch.org", "download-r2.pytorch.org"):
                raise SetupError("GPU用wheelは公式CUDA 13.0版へ固定します")
        elif urlsplit(item["url"]).hostname != "files.pythonhosted.org":
            raise SetupError("pip/Triton Windowsは公式PyPI wheelへ固定します")
        selected[name] = item
    require_secure_torch(selected["torch"]["version"])
    return selected


def _rows(raw, tags, environment):
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_PACKAGES:
        raise SetupError("依存lockには上限内の全wheel一覧が必要です")
    result = {}
    for row in raw:
        if not isinstance(row, dict):
            raise SetupError("依存wheelの記録形式が不正です")
        item = _wheel(row.get("name"), row.get("version"), row.get("url"), row.get("sha256"), tags)
        if item["name"] in result or row.get("filename", item["filename"]) != item["filename"]:
            raise SetupError("依存wheelの名称が重複または不一致です")
        dependencies = row.get("requires_dist", [])
        _requirements(dependencies)  # Reject source/VCS/local dependency references too.
        python_spec = row.get("requires_python", "")
        try:
            if not isinstance(python_spec, str) or python_spec and not SpecifierSet(python_spec).contains(environment["python_full_version"]):
                raise ValueError()
        except (ValueError, InvalidSpecifier) as exc:
            raise SetupError("依存wheelのPython条件を満たしません") from exc
        result[item["name"]] = {**item, "requires_dist": dependencies, "requires_python": python_spec}
    return result


def _validate_plan(rows, selected, roots, environment):
    for name in (*DIRECT, *(('pip',) if 'pip' in rows else ())):
        if name not in rows or any(rows[name][field] != selected[name][field] for field in ("version", "url", "sha256")):
            raise SetupError("pipの取得計画が固定したGPU wheelと一致しません: " + name)
    require_secure_torch(rows["torch"]["version"])
    extras = {name: set() for name in rows}
    pending = list(roots)
    checked = set()
    while pending:
        requirement = pending.pop()
        if not _active(requirement, environment):
            continue
        name = canonicalize_name(requirement.name)
        if name not in rows or requirement.specifier and Version(rows[name]["version"]) not in requirement.specifier:
            raise SetupError("依存lockが要求・推移依存のすべてを満たしていません: " + name)
        new_extras = extras[name] | set(requirement.extras)
        key = name, tuple(sorted(new_extras))
        if key in checked:
            continue
        extras[name] = new_extras
        checked.add(key)
        for child in _requirements(rows[name]["requires_dist"]):
            if _active(child, environment, new_extras):
                # Its parent extra has already been evaluated. Remove that
                # marker before following the dependency's own extra set.
                child.marker = None
                pending.append(child)
    if set(rows) != {name for name, _ in checked}:
        raise SetupError("依存lockに要求の推移関係から確認できないwheelがあります")


def _report(path):
    report = _read_json(path)
    if report.get("version") != "1" or not isinstance(report.get("install"), list):
        raise SetupError("pipのwheel解決reportの版・一覧が不正です")
    rows = []
    for item in report["install"]:
        if not isinstance(item, dict) or item.get("is_yanked") is True:
            raise SetupError("pipの計画に不正または取り下げ済みのwheelがあります")
        try:
            metadata, source = item["metadata"], item["download_info"]
            archive = source["archive_info"]
            digest = _hash(archive["hashes"]["sha256"])
            if archive.get("hash", "sha256=" + digest) != "sha256=" + digest:
                raise ValueError()
            rows.append({"name": metadata["name"], "version": metadata["version"], "url": source["url"],
                         "sha256": digest, "requires_dist": metadata.get("requires_dist", []),
                         "requires_python": metadata.get("requires_python", "")})
        except (KeyError, TypeError, ValueError) as exc:
            raise SetupError("pipの計画にURL・公開hash・名称・版が不足しています") from exc
    return rows


def _bootstrap(raw, rows):
    if not isinstance(raw, dict) or set(raw) - {"setuptools", "wheel"}:
        raise SetupError("bootstrap依存の置換記録が不正です")
    for name, item in raw.items():
        if (not isinstance(item, dict) or set(item) != {"from", "to"} or name not in rows
                or _version(item["to"]) != rows[name]["version"]):
            raise SetupError("bootstrap依存の置換先がlockと一致しません")
        _version(item["from"])
    return raw


def _check_existing(installed, rows, bootstrap):
    for name, version in installed.items():
        if name in rows and Version(version) != Version(rows[name]["version"]):
            if name not in bootstrap or Version(version) != Version(bootstrap[name]["from"]):
                raise SetupError("導入済み依存の版がlockと異なります。既存版を置換しません: " + name)


def install_locked_dependencies(connection, data, folder, run_command, cancelled=lambda: False,
                                progress=lambda **kw: None, log=None):
    """Resolve one complete immutable wheel plan, or resume exactly that plan.

    Call only after pip self-upgrade in a newly owned venv, under the caller's
    setup lock. Existing user environments must use their preservation path.
    """
    if cancelled():
        raise KeyboardInterrupt
    if not isinstance(data, dict) or data.get('torch_index', '').rstrip('/') != TORCH_INDEX:
        raise SetupError("GPU依存の取得構成は公式CUDA 13.0版へ限定します")
    root, python = Path(connection.root).resolve(), Path(connection.python).resolve()
    expected = (root / ".venv/Scripts/python.exe").resolve()
    if python != expected or not python.is_relative_to(root):
        raise SetupError("依存lockは新規導入した選択root内の同一venv専用です")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    log = Path(log) if log is not None else folder / "libraries.log"
    version, installed = _probe(connection, folder, run_command, cancelled, progress, log)
    environment = _environment(version)
    abi = version[:2]
    tags = set(cpython_tags(abi, abis=[f"cp{abi[0]}{abi[1]}"], platforms=["win_amd64"]))
    tags.update(compatible_tags(abi, interpreter=f"cp{abi[0]}{abi[1]}", platforms=["win_amd64"]))
    selected = _selected(data, tags)
    if installed.get("pip") != selected["pip"]["version"]:
        raise SetupError("先に解決済み公式pipへの更新を完了してください")
    requirements_path = root / "requirements.txt"
    try:
        content = requirements_path.read_bytes()
        core = _requirements(content.decode("utf-8-sig").splitlines())
    except (OSError, UnicodeError) as exc:
        raise SetupError("選択coreのrequirementsを確認できません") from exc
    gpu = _requirements(data.get("gpu_requirements"))
    extra = _requirements(data.get("requirements"))
    roots = [*core, *gpu, *extra, *(Requirement(name + "==" + selected[name]["version"]) for name in DIRECT)]
    identity = {"root": str(root).casefold(), "python": str(python).casefold(), "python_abi": list(abi),
                "core_requirements_sha256": sha256(content).hexdigest(), "gpu_requirements": data["gpu_requirements"],
                "requirements": data["requirements"], "direct_packages": selected, "index_url": PYPI_INDEX}
    lock_path, reused = folder / LOCK_NAME, False
    if lock_path.exists():
        lock = _read_json(lock_path)
        if (lock.get("schema_version") != 1 or lock.get("identity") != identity or lock.get("identity_sha256") != _digest(identity)
                or lock.get("packages_sha256") != _digest(lock.get("packages"))):
            raise SetupError("保存された依存lockとroot・ABI・要求・hashが異なります。再解決・上書きしません")
        rows = _rows(lock.get("packages"), tags, environment)
        bootstrap = _bootstrap(lock.get("bootstrap_replacements"), rows)
        reused = True
    else:
        progress(phase="全依存の公式wheelを解決", model="dependency-lock")
        report = _temporary(folder, ".report.json")
        try:
            direct = [name + " @ " + selected[name]["url"] + "#sha256=" + selected[name]["sha256"] for name in DIRECT]
            run_command([python, "-B", "-m", "pip", "--isolated", "install", "--dry-run", "--ignore-installed",
                         "--only-binary=:all:", "--index-url", PYPI_INDEX, "--report", report,
                         "-r", requirements_path, *direct, *data["gpu_requirements"], *data["requirements"]],
                        root, log, cancelled, progress)
            rows = _rows(_report(report), tags, environment)
            _validate_plan(rows, selected, roots, environment)
            if requirements_path.read_bytes() != content:
                raise SetupError("依存解決中にcoreのrequirementsが変更されました")
            bootstrap = {name: {"from": installed[name], "to": rows[name]["version"]}
                         for name in ("setuptools", "wheel") if name in installed and name in rows
                         and installed[name] != rows[name]["version"] and set(installed) <= {"pip", "setuptools", "wheel"}}
            _check_existing(installed, rows, bootstrap)
            atomic_json(lock_path, {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                                   "identity": identity, "identity_sha256": _digest(identity),
                                   "packages": [rows[name] for name in sorted(rows)],
                                   "packages_sha256": _digest([rows[name] for name in sorted(rows)]),
                                   "bootstrap_replacements": bootstrap})
        finally:
            _remove(report)
    _validate_plan(rows, selected, roots, environment)
    _check_existing(installed, rows, bootstrap)
    current_version, current = _probe(connection, folder, run_command, cancelled, progress, log)
    if current_version != version or current != installed:
        raise SetupError("依存計画の確認中に環境が変更されました。インストールしません")
    if requirements_path.read_bytes() != content:
        raise SetupError("導入前にcoreのrequirementsが変更されました。インストールしません")
    missing = [name for name, item in rows.items() if installed.get(name) != item["version"]]
    if missing:
        if cancelled():
            raise KeyboardInterrupt
        requirements = _temporary(folder, ".wheels.txt")
        try:
            requirements.write_text("\n".join(name + " @ " + item["url"] + " --hash=sha256:" + item["sha256"]
                                            for name, item in sorted(rows.items())) + "\n", encoding="utf-8")
            progress(phase="固定した全wheelを導入", model="dependency-lock")
            run_command([python, "-B", "-m", "pip", "--isolated", "install", "--no-deps", "--require-hashes",
                         "--only-binary=:all:", "--index-url", PYPI_INDEX, "-r", requirements], root, log, cancelled, progress)
        finally:
            _remove(requirements)
    _, after = _probe(connection, folder, run_command, cancelled, progress, log)
    if any(after.get(name) != row["version"] for name, row in rows.items()):
        raise SetupError("導入後の全依存の版がlockと一致しません")
    if any(after.get(name) != old for name, old in installed.items() if name not in bootstrap):
        raise SetupError("導入前からある依存の版が変化しました。生成を開始しません")
    run_command([python, "-B", "-m", "pip", "--isolated", "check"], root, log, cancelled, progress)
    return {"scope": "all-transitive-wheel-hashes-v1", "lock": str(lock_path), "lock_reused": reused,
            "packages": len(rows), "installed_or_resumed": missing, "pip_check": True,
            "existing_versions_preserved": not bool(bootstrap), "existing_nonbootstrap_versions_preserved": True,
            "bootstrap_replacements": bootstrap,
            "resolution_egress_firewall": False}
