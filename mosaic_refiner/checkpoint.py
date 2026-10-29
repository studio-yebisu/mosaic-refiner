"""Local atomic job records with an OS-released single-writer lock."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA_VERSION = 1
PIPELINE_VERSION = "cpu-rgb-1"


def digest_file(path: Path) -> str:
    from .native_jobs import selected, call
    if (native := selected()) is not None:
        return call(native, "hash", path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_config(config: dict) -> str:
    data = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def implementation_fingerprint() -> dict:
    """Conservatively invalidate jobs when Python or Rust implementation changes.

    Relative names and normalized line endings keep the record checkout-independent.
    Documentation, fixtures, media and environment secrets are never read here.
    """
    root = Path(__file__).resolve().parent
    files = {path.relative_to(root).as_posix(): hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in sorted(root.rglob("*.py"))}
    native = root.parent / "native"
    if native.is_dir():
        for path in sorted(native.rglob("*")):
            if "target" not in path.relative_to(native).parts and path.is_file() and (path.suffix == ".rs" or path.name in ("Cargo.toml", "Cargo.lock", "cli-schema.json")):
                files["native/" + path.relative_to(native).as_posix()] = hashlib.sha256(
                    path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {"algorithm": "package-python-rust-lf-v2", "files": files, "sha256": digest_config(files)}


def atomic_json(path: Path, data: dict) -> None:
    from .native_jobs import selected, call
    if (native := selected()) is not None:
        call(native, "atomic", path, data)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict:
    from .native_jobs import selected, call
    if (native := selected()) is not None:
        return call(native, "load", path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON objectが必要です: {path.name}")
    return value


@contextmanager
def job_lock(directory: Path):
    """Persistent lock inode, OS lock released on exit/crash. Never unlink it."""
    directory.mkdir(parents=True, exist_ok=True)
    from .native_jobs import selected, lock
    if (native := selected()) is not None:
        with lock(native, directory / ".lock"):
            yield
        return
    with (directory / ".lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("このjobは別のprocessが使用中です") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)


def request_cancel(directory: Path) -> None:
    if not (directory / "manifest.json").is_file():
        raise ValueError("jobのmanifestがありません")
    atomic_json(directory / "cancel.json", {"cancel": True})


def summarize(directory: Path) -> dict:
    from .native_jobs import selected, call
    if (native := selected()) is not None:
        return call(native, "summary", directory)
    data = load_json(directory / "manifest.json")
    return {
        "stage": data.get("stage", "cpu-diagnostic-no-ai"),
        "state": data["state"],
        "job_id": data["job_id"],
        "completed_units": sum(u.get("state") == "complete" for u in data["units"]),
        "total_units": len(data["units"]),
        "output": data.get("output"),
        "last_error": data.get("last_error"),
        "verification": data.get("verification"),
        "integrity_checked": False,
    }
