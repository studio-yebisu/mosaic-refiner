"""Borrowed job-record C ABI, preserving the Python checkpoint API for workers."""
from contextlib import contextmanager
import ctypes as ct
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from .native_core import ROOT, native_source_fingerprint

LIBRARY_NAME = "mosaic_refiner_host.dll" if os.name == "nt" else "libmosaic_refiner_host.so"


@lru_cache(maxsize=3)
def library(mode):
    if mode == "python":
        return None
    directory = ROOT / "mosaic_refiner/_native"
    path, record_path = directory / LIBRARY_NAME, directory / "jobs-build.json"
    if not path.exists() and not record_path.exists():
        if mode == "rust":
            raise ValueError("Rust job core未構築です")
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["source_sha256"] != native_source_fingerprint() or record["library_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Rust job coreのsource/hashが不一致です。明示的に再構築してください")
    lib = ct.CDLL(str(path))
    lib.mr_host_abi_version.restype = ct.c_uint32
    if lib.mr_host_abi_version() != 1:
        raise ValueError("Rust job coreのABIが不一致です")
    lib.mr_host_json.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_size_t)]
    lib.mr_host_json.restype = ct.c_int
    lib.mr_host_free.argtypes = [ct.c_void_p, ct.c_size_t]
    lib.mr_host_lock.argtypes = [ct.c_void_p, ct.c_size_t]
    lib.mr_host_lock.restype = ct.c_void_p
    lib.mr_host_unlock.argtypes = [ct.c_void_p]
    return lib


def selected():
    mode = os.environ.get("MOSAIC_REFINER_HOST", "auto")
    if mode not in ("auto", "rust", "python"):
        raise ValueError("MOSAIC_REFINER_HOSTはauto / rust / pythonです")
    return library(mode)


def call(lib, operation, path, data=None):
    body = json.dumps({"operation": operation, "path": str(path), "data": data}, ensure_ascii=False).encode()
    pointer, length = ct.c_void_p(), ct.c_size_t()
    if lib.mr_host_json(body, len(body), ct.byref(pointer), ct.byref(length)):
        raise ValueError("Rust job core: bufferが不正です")
    try:
        result = json.loads(ct.string_at(pointer, length.value))
    finally:
        lib.mr_host_free(pointer, length.value)
    if not result["ok"]:
        raise ValueError(result["error"])
    return result["data"]


@contextmanager
def lock(lib, path):
    value = str(path).encode()
    handle = lib.mr_host_lock(value, len(value))
    if not handle:
        raise ValueError("このjobは別のprocessが使用中、またはlockを開けません")
    try:
        yield
    finally:
        lib.mr_host_unlock(handle)
