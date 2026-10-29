"""Materialize one CPU-only H3 reference window; never contact the GPU server."""

from fractions import Fraction
import hashlib
import os
from pathlib import Path
import shutil

from .checkpoint import atomic_json, digest_file, implementation_fingerprint
from .geometry import Rect, crop_rgb
from .media import ChunkEncoder, MediaError, MediaInfo, probe, read_exact, rgb_reader, tool_version


def _sample_map(source_size: int, content_size: int, canvas_size: int) -> list[int]:
    # Pixel-centre nearest neighbour, then edge extension into padding.
    return [min(source_size - 1, (2 * min(x, content_size - 1) + 1) * source_size // (2 * content_size))
            for x in range(canvas_size)]


def _canvas_rgb(region: bytes, width: int, xs: list[int], ys: list[int]) -> bytes:
    import numpy as np
    pixels = np.frombuffer(region, np.uint8).reshape(-1, width, 3)
    return pixels[np.asarray(ys)[:, None], np.asarray(xs)[None, :]].tobytes()


def _half_plane(data: bytes, width: int, height: int, channels: int) -> bytes:
    # Exactly the odd pixel centres of the high canvas, including padded pixels.
    import numpy as np
    return np.frombuffer(data, np.uint8).reshape(height, width, channels)[1::2, 1::2].tobytes()


def _mask(request: dict, xs: list[int], ys: list[int], regions=None, entries=None, crop=None) -> bytes:
    import numpy as np
    crop, roi, canvas = crop or request["crop"], request["roi"], request["canvas"]
    feather = request["feather"]
    result = np.zeros((len(ys), len(xs)), np.uint8)
    if entries is not None:
        from .shape_masks import source_alpha
        alpha = source_alpha(entries, Rect(**crop), feather)
        result[:canvas["content_height"], :canvas["content_width"]] = np.rint(255 * alpha[
            np.asarray(ys[:canvas["content_height"]])[:, None],
            np.asarray(xs[:canvas["content_width"]])[None, :]]).astype(np.uint8)
        return result.tobytes()
    px = np.asarray(xs[:canvas["content_width"]])[None, :] + crop["x"]
    py = np.asarray(ys[:canvas["content_height"]])[:, None] + crop["y"]
    for r in regions if regions is not None else (roi,):
        valid = (px >= r["x"]) & (px < r["x"] + r["width"]) & (py >= r["y"]) & (py < r["y"] + r["height"])
        distance = np.minimum(np.minimum(px - r["x"], r["x"] + r["width"] - 1 - px),
                              np.minimum(py - r["y"], r["y"] + r["height"] - 1 - py))
        edge = min(feather, r["width"] // 2, r["height"] // 2)
        value = np.rint(255 * np.clip(distance / edge, 0, 1)).astype(np.uint8) if edge else np.full(valid.shape, 255, np.uint8)
        view = result[:canvas["content_height"], :canvas["content_width"]]
        np.maximum(view, np.where(valid, value, 0), out=view)
    return result.tobytes()


def _flush(path: Path) -> None:
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def _verify_reference(path: Path, width: int, height: int, count: int, expected_rgb_hash: str) -> dict:
    info = probe(path)
    if (info.width, info.height, info.frame_count, Fraction(info.fps), info.audio_streams) != (
            width, height, count, Fraction(24), 0):
        raise MediaError("H3参照動画の寸法/frame数/24fps/音声なしが計画と一致しません")
    digest = hashlib.sha256()
    with rgb_reader(path) as reader:
        for _ in range(count):
            digest.update(read_exact(reader, width * height * 3))
    if digest.hexdigest() != expected_rgb_hash:
        raise MediaError("H3参照動画の復号RGBが入力batchと一致しません")
    return {"pts_exact": True, "full_rgb_decode_exact": True, "rgb_sha256": expected_rgb_hash,
            "frame_count": count, "fps": 24}


def materialize_window(source: Path, plan: dict, output_dir: Path, window_index: int = 0,
                       cancelled=lambda: False) -> dict:
    """Save low/high lossless references, static pixel masks and exact time maps.

    A complete manifest proves CPU input preparation only. It is not a workflow,
    latent noise mask or restored video. Failed/cancelled bundles cannot be resumed;
    callers must choose a fresh directory. No existing output is replaced.
    """
    source, output_dir = source.resolve(), output_dir.resolve()
    if output_dir == source or output_dir in source.parents:
        raise ValueError("入力原本を含むディレクトリへ出力できません")
    if not 0 <= window_index < len(plan["windows"]):
        raise ValueError("window-indexが計画の範囲外です")
    request = plan["windows"][window_index]
    if request["adapter_contract_version"] != 4:
        raise ValueError("H3入力計画の版が一致しません")
    if implementation_fingerprint() != request["implementation"]:
        raise ValueError("計画後に実装が変更されました。計画から再実行してください")
    if digest_file(source) != request["source_sha256"]:
        raise ValueError("計画後に入力が変更されました。計画から再実行してください")
    info = MediaInfo(**plan["source_info"])
    canvas, window = request["canvas"], request["window"]
    width, height = canvas["width"], canvas["height"]
    count = window["padded_model_frames"]
    indices = request["source_frame_indices"]
    dynamic = request.get("frame_regions")
    frame_entries, frame_crops = request.get("frame_entries"), request.get("frame_crops")
    if dynamic is not None and len(dynamic) != count:
        raise ValueError("動的maskのframe数が参照動画と一致しません")
    for mapping in (frame_entries, frame_crops):
        if mapping is not None and len(mapping) != count:
            raise ValueError("追従crop/形状maskのframe数が参照動画と一致しません")
    if len(indices) != count or indices != sorted(indices) or any(
            not window["scene_first_frame"] <= i < window["scene_end_frame_exclusive"] for i in indices):
        raise ValueError("scene内のframe対応表が不正です")
    crop = Rect(**request["crop"])
    crop.validate(info.width, info.height)
    if frame_crops is not None:
        for rect in frame_crops:
            r = Rect(**rect)
            r.validate(info.width, info.height)
            if (r.width, r.height) != (crop.width, crop.height):
                raise ValueError("追従cropの寸法は窓内で固定する必要があります")
    xs = _sample_map(crop.width, canvas["content_width"], width)
    ys = _sample_map(crop.height, canvas["content_height"], height)
    mask = _mask(request, xs, ys, dynamic[0] if dynamic is not None else None,
                 frame_entries[0] if frame_entries is not None else None)
    low_mask = _half_plane(mask, width, height, 1)
    if dynamic is None and (not any(mask) or not any(low_mask)):
        raise ValueError("縮小/featherにより書戻しmaskが空になります。ROI・余白・feather・解像度候補を見直してください")
    versions = {name: tool_version(name) for name in ("ffmpeg", "ffprobe")}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    needed = count * (width * height + width * height // 4) * 4 + 16 * 1024 * 1024
    if shutil.disk_usage(output_dir.parent).free < needed:
        raise MediaError("H3入力bundle用の空きdiskが保守的見積もりを下回りました")
    output_dir.mkdir(exist_ok=False)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 1, "stage": "h3-cpu-input-bundle", "state": "preparing",
        "ready_to_submit": False, "executable_workflow": False, "ai_restoration": False,
        "profile": plan["profile"], "request": request, "source_info": info.to_dict(),
        "artifacts": {}, "unresolved": plan["unresolved"],
        "mask_contract": "dynamic 24fps lossless RGB masks; padding=0" if dynamic is not None else
                         "static PGM uint8: 0=keep, 255=write; padding=0; not a latent noise mask",
        "resize": "high: pixel-centre nearest then edge pad; low: high pixel (2*x+1,2*y+1)",
        "source_write_mask_is_authoritative": True,
        **versions,
    }
    atomic_json(manifest_path, manifest)
    encoders = []
    try:
        stages = (("high", width, height, mask), ("low", width // 2, height // 2, low_mask))
        hashes = {name: hashlib.sha256() for name, *_ in stages}
        nonzero = {"high-mask": 0, "low-mask": 0}
        for name, w, h, pixels in stages:
            mask_path = output_dir / f"{name}-write-mask.pgm"
            with mask_path.open("xb") as stream:
                stream.write(f"P5\n{w} {h}\n255\n".encode("ascii") + pixels)
                stream.flush()
                os.fsync(stream.fileno())
            manifest["artifacts"][mask_path.name] = {"sha256": digest_file(mask_path),
                "bytes": mask_path.stat().st_size, "width": w, "height": h, "repeat_frames": count,
                "nonzero_pixels": sum(value > 0 for value in pixels)}
            target = output_dir / f"{name}-reference.nut.partial"
            stage_info = MediaInfo(w, h, count, "24", "1/24", 0, "rgb24", "unknown", "unknown")
            encoders.append((name, ChunkEncoder(target, stage_info)))
            if dynamic is not None:
                encoders.append((name + "-mask", ChunkEncoder(output_dir / f"{name}-mask.nut.partial", stage_info)))
                hashes[name + "-mask"] = hashlib.sha256()
        with rgb_reader(source, indices[0], info) as reader:
            current = indices[0] - 1
            high = low = None
            previous_crop = None
            for model_index, index in enumerate(indices):
                if cancelled():
                    raise KeyboardInterrupt
                current_crop = Rect(**frame_crops[model_index]) if frame_crops else crop
                changed = index != current or current_crop != previous_crop
                while current < index:
                    frame = read_exact(reader, info.width * info.height * 3)
                    current += 1
                if changed:
                    high = _canvas_rgb(crop_rgb(frame, info.width, info.height, current_crop), crop.width, xs, ys)
                    low = _half_plane(high, width, height, 3)
                previous_crop = current_crop
                if dynamic is not None:
                    import numpy as np
                    high_mask = _mask(request, xs, ys, dynamic[model_index],
                                      frame_entries[model_index] if frame_entries is not None else None,
                                      frame_crops[model_index] if frame_crops is not None else None)
                    low_mask = _half_plane(high_mask, width, height, 1)
                    mask_rgb = {"high-mask": np.repeat(np.frombuffer(high_mask, np.uint8), 3).tobytes(),
                                "low-mask": np.repeat(np.frombuffer(low_mask, np.uint8), 3).tobytes()}
                    for name in nonzero:
                        nonzero[name] += np.count_nonzero(np.frombuffer(mask_rgb[name], np.uint8)) // 3
                for name, encoder in encoders:
                    pixels = mask_rgb[name] if name.endswith("-mask") else high if name == "high" else low
                    encoder.write(pixels)
                    hashes[name].update(pixels)
        if dynamic is not None and not all(nonzero.values()):
            raise ValueError("動的な書戻しmaskが全frameで空になりました")
        while encoders:
            _, encoder = encoders.pop()
            encoder.close()
        for name, w, h, _ in stages:
            partial = output_dir / f"{name}-reference.nut.partial"
            qa = _verify_reference(partial, w, h, count, hashes[name].hexdigest())
            _flush(partial)
            target = output_dir / f"{name}-reference.nut"
            os.replace(partial, target)
            manifest["artifacts"][target.name] = {"sha256": digest_file(target),
                "bytes": target.stat().st_size, "width": w, "height": h, "verification": qa}
            if dynamic is not None:
                partial = output_dir / f"{name}-mask.nut.partial"
                qa = _verify_reference(partial, w, h, count, hashes[name + "-mask"].hexdigest())
                _flush(partial)
                target = output_dir / f"{name}-mask.nut"
                os.replace(partial, target)
                manifest["artifacts"][target.name] = {"sha256": digest_file(target), "bytes": target.stat().st_size,
                                                     "width": w, "height": h, "verification": qa}
        fps, tb = Fraction(info.fps), Fraction(info.time_base)
        mapping = output_dir / "frame-map.json"
        atomic_json(mapping, {"model_time_base": "1/24", "source_time_base": info.time_base,
            "window_start_seconds": request["source_timeline"]["window_start_seconds"] if request.get("source_timeline") else
                                    str(Fraction(window["scene_first_frame"], 1) / fps + Fraction(window["model_first_frame"], 24)),
            "trim_to_model_frames": window["valid_model_frames"], "frames": [
                {"model_pts": i, "source_frame": index,
                 "source_pts": info.frame_pts[index] if info.frame_pts else int(Fraction(index, 1) / fps / tb),
                 "padding": i >= window["valid_model_frames"],
                 "crop": frame_crops[i] if frame_crops is not None else request["crop"]}
                for i, index in enumerate(indices)]})
        manifest["artifacts"][mapping.name] = {"sha256": digest_file(mapping), "bytes": mapping.stat().st_size}
        if digest_file(source) != request["source_sha256"]:
            raise MediaError("準備中に入力原本が変更されました")
        if implementation_fingerprint() != request["implementation"]:
            raise MediaError("準備中に実装が変更されました")
        manifest["state"] = "complete"
        atomic_json(manifest_path, manifest)
    except BaseException as exc:
        for _, encoder in encoders:
            encoder.close(abort=True)
        manifest.update(state="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed", last_error=str(exc))
        atomic_json(manifest_path, manifest)
        raise
    return {"stage": manifest["stage"], "state": "complete", "window_index": window_index,
            "manifest": str(manifest_path), "ready_to_submit": False, "ai_restoration": False,
            "frame_count": count, "trim_to_model_frames": window["valid_model_frames"]}
