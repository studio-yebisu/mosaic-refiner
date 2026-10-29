"""Compatibility entry point for the explicitly built Rust application host."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HOST_NAME = "mosaic-refiner.exe" if os.name == "nt" else "mosaic-refiner"


def invoke_if_available(argv):
    mode = os.environ.get("MOSAIC_REFINER_HOST", "auto")
    if mode not in ("auto", "rust", "python"):
        raise ValueError("MOSAIC_REFINER_HOSTはauto / rust / pythonです")
    if mode == "python":
        return None
    directory = ROOT / "mosaic_refiner/_native"
    host, record_path = directory / HOST_NAME, directory / "host-build.json"
    if not host.exists() and not record_path.exists():
        if mode == "rust":
            raise ValueError("Rustアプリ未構築です。scripts/build_native.pyを実行してください")
        return None
    from .native_core import native_source_fingerprint
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (hashlib.sha256(host.read_bytes()).hexdigest() != record["host_sha256"]
            or native_source_fingerprint() != record["source_sha256"]
            or hashlib.sha256((ROOT / "mosaic_refiner/web.html").read_bytes().replace(b"\r\n", b"\n")).hexdigest() != record["ui_sha256"]):
        raise ValueError("Rustアプリのsource/実行ファイル版が異なります。明示的に再構築してください")
    return subprocess.call([str(host), "--project-root", str(ROOT), "--worker-python", sys.executable, *argv])
