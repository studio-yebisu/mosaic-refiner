"""CPU request preparation and read-only checks for the canonical H3 server.

Ref2Video is a condition, not an inpainting mask. This module cannot submit,
cancel, unload, restart, download, or change the runtime. Its JSON is not a WF.
"""

from dataclasses import asdict
from bisect import bisect_left
from fractions import Fraction
import json
from math import ceil, sqrt, isfinite
from pathlib import Path
from urllib.error import URLError
from .runtime_http import urlopen

from .checkpoint import digest_config, digest_file, implementation_fingerprint
from .geometry import Rect
from .media import probe
from .planning import model_windows, model_windows_timeline, plan_cfr


from .runtime_settings import RUNTIME
CANONICAL_ENDPOINT = RUNTIME.endpoint
from .h3_profile import assets as public_assets, loader_names
MODEL_CANDIDATE = loader_names()['dit']
MODEL_EXPECTED_SHA256 = public_assets()['dit'].sha256
UNRESOLVED = [
    "物理12GiBでの初回load/conditioning/VAE/sampling peak実測",
    "今回の変更後の画質・継ぎ目・忠実性の利用者レビュー",
    "クリーンPCからのモデル全量取得・H3 node/kernel導入・実生成",
]


def fit_canvas(width: int, height: int, max_pixels: int = 262144, minimum_edge: int = 64,
               content_scale: float = 1.0) -> dict:
    if width < 1 or height < 1 or minimum_edge < 64 or minimum_edge % 64 or max_pixels < minimum_edge**2:
        raise ValueError("寸法は正数、最小辺は64の倍数、max-pixelsは最小canvas面積以上が必要です")
    if not isfinite(content_scale) or not 0 < content_scale <= 2:
        raise ValueError("再生成の縮尺は0より大きく2以下です")
    scale = min(content_scale, sqrt(max_pixels / (width * height)))
    while True:
        cw, ch = max(1, int(width * scale)), max(1, int(height * scale))
        w, h = max(minimum_edge, ceil(cw / 64) * 64), max(minimum_edge, ceil(ch / 64) * 64)
        if w * h <= max_pixels:
            return {"content_width": cw, "content_height": ch, "width": w, "height": h,
                    "pad_right": w - cw, "pad_bottom": h - ch,
                    "low_width": w // 2, "low_height": h // 2,
                    "source_to_content_x": str(Fraction(cw, width)),
                    "source_to_content_y": str(Fraction(ch, height)),
                    "padding_mode": "edge", "alignment": 64, "measured_12gib": False}
        scale *= 0.9


def prepare_request(source: Path, roi: Rect, context_seconds: int = 5,
                    unit_seconds: Fraction = Fraction(15), overlap_seconds: Fraction = Fraction(1),
                    scene_cuts: tuple[int, ...] = (), max_pixels: int = 262144,
                    seed: int = 0, prompt: str = "Preserve the scene and motion in <Video 1>.",
                    crop_margin: int = 0, feather: int = 0, allow_timeline: bool = False, source_info=None,
                    content_scale: float = 1.0) -> dict:
    info = source_info if source_info is not None else probe(source, allow_timeline=allow_timeline)
    roi.validate(info.width, info.height)
    if crop_margin < 0:
        raise ValueError("crop-marginは0以上です")
    if not 0 <= feather <= min(roi.width, roi.height) // 2:
        raise ValueError("featherは0以上、ROI短辺の半分以下です")
    left, top = max(0, roi.x - crop_margin), max(0, roi.y - crop_margin)
    crop = Rect(left, top, min(info.width, roi.x + roi.width + crop_margin) - left,
                min(info.height, roi.y + roi.height + crop_margin) - top)
    if seed < 0 or seed >= 2**63:
        raise ValueError("seedは0以上2^63未満です")
    if not prompt.strip():
        raise ValueError("promptは空にできません")
    source_hash = digest_file(source)
    windows = (model_windows_timeline(info.frame_pts, info.frame_durations, info.time_base,
               context_seconds, overlap_seconds, scene_cuts) if info.frame_pts else
               model_windows(info.frame_count, Fraction(info.fps), context_seconds, overlap_seconds, scene_cuts))
    canvas = fit_canvas(crop.width, crop.height, max_pixels, content_scale=content_scale)
    implementation = implementation_fingerprint()
    requests = []
    timeline_times = [p * Fraction(info.time_base) for p in info.frame_pts]
    for window in windows:
        base = {"source_sha256": source_hash, "window": asdict(window), "roi": asdict(roi),
                "crop": asdict(crop), "feather": feather, "implementation": implementation,
                "canvas": canvas, "prompt": prompt, "model_expected_sha256": MODEL_EXPECTED_SHA256,
                "lora_profile": "unlocked_v2", "upscaler": "bicubic-latent-x2", "job_seed": seed,
                "adapter_contract_version": 4}
        key = digest_config(base)
        indices = [window.source_frame(i, Fraction(info.fps)) for i in range(window.padded_model_frames)]
        timeline = None
        if info.frame_pts:
            times = timeline_times
            first, end = window.scene_first_frame, window.scene_end_frame_exclusive
            indices = []
            for index in range(window.padded_model_frames):
                instant = times[first] + Fraction(window.model_first_frame + min(index, window.valid_model_frames - 1), 24)
                right = min(end - 1, bisect_left(times, instant, first, end))
                nearest = right if right == first or abs(times[right] - instant) <= abs(times[right - 1] - instant) else right - 1
                indices.append(nearest)
            timeline = {"scene_start_seconds": str(times[first]),
                        "window_start_seconds": str(times[first] + Fraction(window.model_first_frame, 24))}
        requests.append({**base, "conditioning_key": key, "window_seed": int(key[:15], 16),
                         "source_frame_indices": indices, "source_timeline": timeline,
                         "trim_to_model_frames": window.valid_model_frames,
                         "write_mask": {"space": "source", "rectangle": asdict(roi), "outside": 0,
                                        "inside": 1, "padding": 0, "feather": feather,
                                        "latent_noise_mask_built": False}})
    return {
        "stage": "h3-request-plan-only", "executable_workflow": False, "ready_to_submit": False,
        "endpoint": CANONICAL_ENDPOINT, "source_info": info.to_dict(),
        "model_candidate": MODEL_CANDIDATE, "weights_locally_verified": False,
        "profile": {"context_seconds": context_seconds, "model_fps": 24, "lora_profile": "unlocked_v2",
                    "guided_upscaler_lora_still_enabled": False, "sampling": "low20-spatial-x2-high10",
                    "target_vram_gib": 12, "worker_peak_budget_gib": 10, "budget_is_assumption": True,
                    "fallback_context_seconds": [c for c in (15, 10, 5) if c <= context_seconds]},
        "save_plan": plan_cfr(info.frame_count, Fraction(info.fps), unit_seconds, context_seconds).to_dict(),
        "windows": requests, "unresolved": UNRESOLVED,
    }


def inspect_runtime() -> dict:
    RUNTIME.validate()
    def get(path):
        with urlopen(CANONICAL_ENDPOINT + path, timeout=10) as response:
            return json.load(response)

    try:
        queue = get("/queue")
        stats = get("/system_stats")
        nodes = {}
        for name in ("MiniMaxH3ReferenceToVideo", "MiniMaxH3FunControlNetApply", "MosaicH3LatentUpscale2x"):
            data = get("/object_info/" + name)
            node = data.get(name, {})
            nodes[name] = {
                "available": bool(node),
                "required_inputs": sorted(node.get("input", {}).get("required", {})),
                "optional_inputs": sorted(node.get("input", {}).get("optional", {})),
            }
        devices = [{key: device.get(key) for key in ("name", "type", "vram_total", "vram_free")}
                   for device in stats.get("devices", [])]
        return {"endpoint": CANONICAL_ENDPOINT, "read_only": True,
                "running": len(queue.get("queue_running", [])), "pending": len(queue.get("queue_pending", [])),
                "devices": devices, "nodes": nodes, "ready_to_submit": False,
                "process_root_verified_by_this_command": False, "unresolved": UNRESOLVED}
    except (URLError, TimeoutError, ValueError) as exc:
        raise ValueError(f"選択したComfyUI APIを読み取れません。初回セットアップと起動を確認してください: {exc}") from exc
