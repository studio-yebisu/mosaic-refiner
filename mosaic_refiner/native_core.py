"""Rust RGB kernels with a measured Python reference; never build on startup."""
import ctypes as ct
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

ABI_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
LIBRARY_NAME = ("mosaic_refiner_core.dll" if os.name == "nt" else
                "libmosaic_refiner_core.dylib" if sys.platform == "darwin" else "libmosaic_refiner_core.so")


def native_source_fingerprint(root=ROOT):
    files = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in sorted((root / "native").rglob("*"))
        if "target" not in p.relative_to(root / "native").parts and p.is_file()
        and (p.suffix == ".rs" or p.name in ("Cargo.toml", "Cargo.lock", "cli-schema.json"))}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _Rect(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in ("x", "y", "width", "height")]


class _Window(ct.Structure):
    _fields_ = [("pixels", ct.c_void_p), ("bytes", ct.c_size_t),
                ("width", ct.c_size_t), ("height", ct.c_size_t),
                ("content_width", ct.c_size_t), ("content_height", ct.c_size_t),
                ("crop_x", ct.c_int64), ("crop_y", ct.c_int64),
                ("crop_width", ct.c_size_t), ("crop_height", ct.c_size_t), ("weight", ct.c_double)]


class _Statistics(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in (
        "changed_channels", "outside_changed_channels", "masked_channels", "masked_abs_error_sum",
        "masked_channels_over8", "masked_channels_over24")] + [("first_outside_channel", ct.c_int64)]


def _mode(engine=None):
    mode = engine or os.environ.get("MOSAIC_REFINER_CORE", "auto")
    if mode not in ("auto", "python", "rust"):
        raise ValueError("MOSAIC_REFINER_COREはauto / python / rustです")
    return mode


@lru_cache(maxsize=3)
def _load(mode):
    if mode == "python":
        return None, {"engine": "python", "reason": "explicit_reference"}
    directory = ROOT / "mosaic_refiner" / "_native"
    path, manifest = directory / LIBRARY_NAME, directory / "build.json"
    if not path.is_file() and not manifest.is_file():
        if mode == "rust":
            raise ValueError("Rust core未構築です。scripts/build_native.pyを実行してください")
        return None, {"engine": "python", "reason": "native_core_not_built"}
    if not path.is_file() or not manifest.is_file():
        raise ValueError("Rust coreの構築記録が不完全です。明示的に再構築してください")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if record.get("library_sha256") != digest or record.get("source_sha256") != native_source_fingerprint():
        raise ValueError("Rust coreのhash/ソース版が異なります。scripts/build_native.pyで再構築してください")
    library = ct.CDLL(str(path))  # CDLL releases the GIL during numeric work.
    library.mr_abi_version.restype = ct.c_uint32
    if library.mr_abi_version() != ABI_VERSION or record.get("abi_version") != ABI_VERSION:
        raise ValueError("Rust coreのABI版が一致しません")
    library.mr_composite_rect.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_size_t,
        _Rect, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_size_t]
    library.mr_composite_windows.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_size_t,
        _Rect, ct.c_void_p, ct.c_size_t, ct.POINTER(_Window), ct.c_size_t, ct.c_void_p, ct.c_size_t]
    library.mr_compare_rgb.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_size_t,
        _Rect, ct.c_void_p, ct.c_size_t, ct.POINTER(_Statistics)]
    for name in ("mr_composite_rect", "mr_composite_windows", "mr_compare_rgb"):
        getattr(library, name).restype = ct.c_int32
    return library, {"engine": "rust", "abi_version": ABI_VERSION,
        "source_sha256": record["source_sha256"], "library_sha256": digest,
        "rustc": record["rustc"], "profile": "release"}


def identity(engine=None):
    return dict(_load(_mode(engine))[1])


def _check(status):
    if status:
        message = {1: "寸法/座標", 2: "buffer長", 3: "pointer", 4: "window重み", 5: "mask"}.get(status, "ABI")
        raise ValueError(f"Rust core: {message}が無効です（{status}）")


def _pixels(frame, width, height):
    import numpy as np
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("RGB寸法は正の整数です")
    array = np.frombuffer(frame, np.uint8)
    if array.size != width * height * 3:
        raise ValueError("RGB frameのbyte数が一致しません")
    return array.reshape(height, width, 3)


def _rect(roi, width, height):
    if any(type(v) is not int for v in (roi.x, roi.y, roi.width, roi.height)):
        raise ValueError("ROI座標は整数です")
    roi.validate(width, height)
    return _Rect(roi.x, roi.y, roi.width, roi.height)


def _alpha(alpha, roi):
    import numpy as np
    array = np.asarray(alpha)
    if array.shape != (roi.height, roi.width) or array.dtype != np.float32:
        raise ValueError("maskはROI寸法のfloat32配列です")
    return np.ascontiguousarray(array)


def composite_rectangle_if_available(frame, width, height, roi, restored, feather):
    library, _ = _load(_mode())
    if library is None:
        return None
    source, result = _pixels(frame, width, height), _pixels(restored, roi.width, roi.height)
    box = _rect(roi, width, height)
    output = bytearray(source.size)
    destination = (ct.c_uint8 * len(output)).from_buffer(output)
    _check(library.mr_composite_rect(source.ctypes.data, source.size, width, height, box,
        result.ctypes.data, result.size, feather, destination, len(output)))
    return output


def compose_frame(frame, width, height, roi, alpha, windows, engine=None):
    """windows carry pixels, canvas, crop and positive ordered temporal weight."""
    import numpy as np
    source, box = _pixels(frame, width, height), _rect(roi, width, height)
    alpha = _alpha(alpha, roi)
    library, _ = _load(_mode(engine))
    if not 1 <= len(windows) <= 64:
        raise ValueError("合成windowは1〜64件です")
    arrays, descriptors = [], []
    total = 0.0
    for window in windows:
        canvas, crop, weight = window["canvas"], window["crop"], float(window["weight"])
        values = [canvas[k] for k in ("width", "height", "content_width", "content_height")]
        values += [crop[k] for k in ("x", "y", "width", "height")]
        if any(type(v) is not int or abs(v) > 2**31 - 1 for v in values):
            raise ValueError("canvas/cropは範囲内の整数です")
        cw, ch, iw, ih = values[:4]
        if not (0 < iw <= cw and 0 < ih <= ch and crop["width"] > 0 and crop["height"] > 0):
            raise ValueError("canvas/crop寸法が無効です")
        pixels = np.asarray(window["pixels"])
        if pixels.dtype != np.uint8 or pixels.shape != (ch, cw, 3):
            raise ValueError("生成RGBの寸法/dtypeがcanvasと異なります")
        arrays.append(np.ascontiguousarray(pixels))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("window重みは有限の正値です")
        total += weight
        descriptors.append(_Window(arrays[-1].ctypes.data, arrays[-1].size, cw, ch, iw, ih,
                                   crop["x"], crop["y"], crop["width"], crop["height"], weight))
    if not np.isfinite(total):
        raise ValueError("window重み総和が非有限です")
    if library is not None:
        packed = (_Window * len(descriptors))(*descriptors)
        output = bytearray(source.size)
        destination = (ct.c_uint8 * len(output)).from_buffer(output)
        _check(library.mr_composite_windows(source.ctypes.data, source.size, width, height, box,
            alpha.ctypes.data, alpha.size, packed, len(packed), destination, len(output)))
        return output
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("maskは有限の0〜1です")
    restored = np.zeros((roi.height, roi.width, 3), dtype=np.float64)
    for window, pixels in zip(windows, arrays):
        canvas, crop = window["canvas"], window["crop"]
        xs = np.clip(((np.arange(roi.width) + roi.x - crop["x"] + .5) *
                      canvas["content_width"] / crop["width"]).astype(int), 0, canvas["content_width"] - 1)
        ys = np.clip(((np.arange(roi.height) + roi.y - crop["y"] + .5) *
                      canvas["content_height"] / crop["height"]).astype(int), 0, canvas["content_height"] - 1)
        restored += pixels[ys[:, None], xs[None, :]] * window["weight"]
    output = source.copy()
    old = output[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]
    old[:] = np.rint(old + alpha[..., None] * (restored / total - old)).clip(0, 255).astype(np.uint8)
    return output.tobytes()


def compare_frame(source, result, width, height, alpha=None, roi=None, engine=None):
    """One traversal for exterior/holes equality and masked numeric metrics."""
    import numpy as np
    a, b = _pixels(source, width, height), _pixels(result, width, height)
    if roi is None:
        if alpha is not None:
            from .geometry import Rect
            roi = Rect(0, 0, width, height)
        else:
            box, mask = _Rect(0, 0, 0, 0), None
    if roi is not None:
        box, mask = _rect(roi, width, height), _alpha(alpha, roi)
    library, _ = _load(_mode(engine))
    if library is not None:
        stats = _Statistics()
        _check(library.mr_compare_rgb(a.ctypes.data, b.ctypes.data, a.size, width, height, box,
            mask.ctypes.data if mask is not None else None, mask.size if mask is not None else 0, ct.byref(stats)))
        return {"changed": bool(stats.changed_channels), "outside_equal": stats.outside_changed_channels == 0,
                "masked_channels": stats.masked_channels, "masked_abs_error_sum": stats.masked_abs_error_sum,
                "masked_channels_over8": stats.masked_channels_over8,
                "masked_channels_over24": stats.masked_channels_over24}
    active = np.zeros((height, width), dtype=bool)
    if mask is not None:
        if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
            raise ValueError("maskは有限の0〜1です")
        active[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width] = mask > 0
    diff = np.abs(a[active].astype(np.int16) - b[active].astype(np.int16))
    return {"changed": not np.array_equal(a, b), "outside_equal": np.array_equal(a[~active], b[~active]),
            "masked_channels": int(diff.size), "masked_abs_error_sum": int(diff.sum()),
            "masked_channels_over8": int(np.count_nonzero(diff > 8)),
            "masked_channels_over24": int(np.count_nonzero(diff > 24))}
