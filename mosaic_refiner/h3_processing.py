"""Lazy window generation and rational source-time overlap composition."""
from fractions import Fraction
from bisect import bisect_left, bisect_right
from pathlib import Path
from threading import Event
import time

from .checkpoint import atomic_json, digest_file, load_json
from .bounded_stream import BoundedStream, interval_overlap_seconds
from .h3_adapter import prepare_request, fit_canvas
from .detection import union
from .follow_crop import follow_crop_path
from .h3_inputs import materialize_window
from .h3_runtime import H3Runtime
from .media import read_exact, rgb_reader
from .processing_options import resolve_options
from .native_core import compose_frame, identity as core_identity


class H3Processor:
    def __init__(self, source, job_dir, roi, feather=8, context_seconds=5,
                 scene_cuts=(), crop_margin=32, max_pixels=262144, seed=0,
                 prompt="Remove remaining pixelation and reconstruct natural fine detail in the restored region of <Video 1>. Follow the restored reference, maintaining the scene, colors, structure and continuous motion.",
                 progress=lambda **kw: None, runtime=None, schedule=None, source_info=None,
                 prefetch=None, processing_options=None):
        self.source, self.job_dir = Path(source).resolve(), Path(job_dir).resolve()
        options = resolve_options(processing_options)
        content_scale = options.render_scale_percent / 100 if processing_options is not None else 1.0
        if processing_options is not None:
            crop_margin, max_pixels = options.crop_margin, options.max_pixels
        self.plan = prepare_request(self.source, roi, context_seconds=context_seconds,
                                    scene_cuts=scene_cuts, crop_margin=crop_margin,
                                    feather=feather, max_pixels=max_pixels, seed=seed, prompt=prompt,
                                    allow_timeline=True, source_info=source_info, content_scale=content_scale)
        self.schedule = schedule
        if schedule is not None:
            width, height = self.plan["source_info"]["width"], self.plan["source_info"]["height"]
            source = self.plan["source_info"]
            fps = Fraction(source["fps"])
            times = [p * Fraction(source["time_base"]) for p in source["frame_pts"]]
            paths = {}
            for request in self.plan["windows"]:
                window = request["window"]
                first, end = window["scene_first_frame"], window["scene_end_frame_exclusive"]
                if first not in paths:
                    paths[first] = follow_crop_path(schedule, width, height, first, end, crop_margin)
            for request in self.plan["windows"]:
                boxes = [box for frame in request["source_frame_indices"] for box in schedule.at(frame)]
                request["inactive"] = not boxes
                if not boxes:
                    continue
                stable = union(boxes)
                request["roi"] = {"x": stable.x, "y": stable.y, "width": stable.width, "height": stable.height}
                x, y = max(0, stable.x - crop_margin), max(0, stable.y - crop_margin)
                request["crop"] = {"x": x, "y": y,
                    "width": min(width, stable.x + stable.width + crop_margin) - x,
                    "height": min(height, stable.y + stable.height + crop_margin) - y}
                request["frame_regions"] = [[{"x": b.x, "y": b.y, "width": b.width, "height": b.height}
                                            for b in schedule.at(frame)] for frame in request["source_frame_indices"]]
                request["frame_entries"] = [list(schedule.entries_at(frame)) for frame in request["source_frame_indices"]]
                path = paths[request["window"]["scene_first_frame"]]
                if path:
                    follow = path[request["source_frame_indices"][0]]
                    old_canvas = fit_canvas(request["crop"]["width"], request["crop"]["height"], max_pixels,
                                            minimum_edge=256, content_scale=content_scale)
                    new_canvas = fit_canvas(follow["width"], follow["height"], max_pixels,
                                            minimum_edge=256, content_scale=content_scale)
                    if new_canvas["width"] * new_canvas["height"] <= .85 * old_canvas["width"] * old_canvas["height"]:
                        request["static_crop"] = request["crop"]
                        request["crop"] = follow
                        request["frame_crops"] = [path[frame] for frame in request["source_frame_indices"]]
                        start_time = (Fraction(request["source_timeline"]["window_start_seconds"])
                                      if request["source_timeline"] else Fraction(request["window"]["scene_first_frame"], 1)/fps + Fraction(request["window"]["model_first_frame"], 24))
                        end_time = start_time + Fraction(request["window"]["valid_model_frames"], 24)
                        if source["frame_pts"]:
                            first_frame, end_frame = bisect_left(times, start_time), bisect_left(times, end_time)
                        else:
                            first_frame, end_frame = int(start_time*fps), min(source["frame_count"], int(end_time*fps)+1)
                        request["source_frame_crops"] = {str(frame): path[frame] for frame in range(first_frame, end_frame) if frame in path}
                        request["crop_method"] = "scene-local-fixed-size-follow-v1"
                    else:
                        request["crop_method"] = "stable-union"
                request["feather"] = min(feather, stable.width // 2, stable.height // 2)
        # Keep the validated minimum canvas at 256 pixels on both axes.
        # Extend padding, retaining native content scale instead of enlarging ROI.
        for request in self.plan["windows"]:
            crop = request["crop"]
            request["canvas"] = fit_canvas(crop["width"], crop["height"], max_pixels,
                                           minimum_edge=256, content_scale=content_scale)
            request["processing_options"] = options.to_dict()
            request["lora_profile"] = options.lora_profile
        self.plan["profile"]["lora_profile"] = options.lora_profile
        self.plan["profile"]["processing_options"] = options.summary()
        self.plan["profile"]["sampling"] = f"low{options.summary()['low_steps']}-spatial-x2-high{options.summary()['high_steps']}"
        self.info = self.plan["source_info"]
        self.fps = Fraction(self.info["fps"])
        self.runtime = runtime or H3Runtime(processing_options=options)
        self.ai_backend = bool(getattr(self.runtime, "ai_backend", False))
        self.prefetch = self.ai_backend if prefetch is None else bool(prefetch)
        self.stop = Event()
        self.prepared_stream = self.generated_stream = None
        self.generated_paths = {}
        self.stage_events = []
        self.generation_busy = False
        self.closed = False
        self.progress = progress
        self.cache = {}
        self.roi, self.feather = roi, feather
        self.scenes = {}
        for request in self.plan["windows"]:
            first = request["window"]["scene_first_frame"]
            self.scenes.setdefault(first, []).append(request)
        self.scene_starts = sorted(self.scenes)
        self.window_starts = {key: [(Fraction(r["source_timeline"]["window_start_seconds"]) if r["source_timeline"]
                                   else Fraction(key, 1)/self.fps + Fraction(r["window"]["model_first_frame"],24)) for r in requests]
                              for key, requests in self.scenes.items()}

    def fingerprint(self):
        return {"runtime": self.runtime.fingerprint(), "requests": self.plan["windows"],
                "cpu_core": core_identity(),
                "bounded_pipeline": {"enabled": self.prefetch, "prepared_capacity": 1, "generated_capacity": 1},
                "schedule": self.schedule.to_dict() if self.schedule else None}

    def _prepare_window(self, request, cancelled):
        index = request["window"]["index"]
        base = self.job_dir / "windows" / f"{index:06d}"
        bundle, generated_dir = base / "input", base / "generation"
        result = generated_dir / "generated.mp4"
        receipt = generated_dir / "result.json"
        intact = False
        if receipt.is_file() and result.is_file():
            data = load_json(receipt)
            intact = data.get("state") == "complete" and data.get("sha256") == digest_file(result)
        bundle_intact = False
        if (bundle / "manifest.json").is_file():
            data = load_json(bundle / "manifest.json")
            bundle_intact = (data.get("state") == "complete" and data.get("request") == request
                             and all((bundle / name).is_file() and digest_file(bundle / name) == artifact["sha256"]
                                     for name, artifact in data.get("artifacts", {}).items()))
            attempted = any(bundle.glob("restored-*-reference*.nut"))
            if attempted:
                from .restoration_models import BASIC_PROFILE
                restoration = load_json(bundle / "restoration.json") if (bundle / "restoration.json").is_file() else {}
                bundle_intact = bundle_intact and (restoration.get("state") == "complete"
                    and restoration.get("profile") == BASIC_PROFILE
                    and all((bundle / name).is_file() and digest_file(bundle / name) == value["sha256"]
                            for name,value in restoration.get("artifacts",{}).items()))
        # Preserve a corrupt completed result, or incomplete input. A complete
        # prepared-only bundle can be reused after cancellation without decoding again.
        if not intact and (not bundle_intact or receipt.exists()):
            if bundle.exists():
                # Preserve failed/corrupt input and result rather than overwriting.
                base.rename(base.with_name(base.name + f"-incomplete-{time.time_ns()}"))
            self.progress(preparing_window=index+1)
            started = time.monotonic()
            materialize_window(self.source, self.plan, bundle, index, cancelled)
            self.stage_events.append({"stage": "prepare", "window": index, "start": started, "end": time.monotonic()})
            self.progress(prepared_window=index+1, preparing_window=None)
        return {"request": request, "bundle": bundle, "directory": generated_dir,
                "result": result if intact else None}

    def _generate_prepared(self, prepared, cancelled):
        request, bundle, directory = prepared["request"], prepared["bundle"], prepared["directory"]
        result = prepared["result"]
        if result is None and (directory / "submission.json").exists() and hasattr(self.runtime, "recover"):
            result = self.runtime.recover(bundle, directory, cancelled, self.progress)
            if result is None:
                base = bundle.parent
                base.rename(base.with_name(base.name + f"-incomplete-{time.time_ns()}"))
                materialize_window(self.source, self.plan, bundle, request["window"]["index"], cancelled)
        if result is None:
            if cancelled():
                raise KeyboardInterrupt
            self.generation_busy = True
            self.progress(window=request["window"]["index"]+1,
                          window_total=len([r for r in self.plan["windows"] if not r.get("inactive")]),
                          phase="H3の入力を準備")
            started = time.monotonic()
            try:
                result = self.runtime.generate(bundle, directory, cancelled, self.progress)
            finally:
                self.generation_busy = False
                self.stage_events.append({"stage": "generate", "window": request["window"]["index"],
                                          "start": started, "end": time.monotonic()})
        if cancelled():
            raise KeyboardInterrupt
        return request["window"]["index"], result

    def start(self, first_frame, cancelled, end_frame=None):
        if not self.prefetch or self.generated_stream is not None:
            return
        first_frame = min(first_frame, self.info["frame_count"]-1)
        start = (self.info["frame_pts"][first_frame] * Fraction(self.info["time_base"])
                 if self.info["frame_pts"] else Fraction(first_frame, 1)/self.fps)
        end_frame = self.info["frame_count"] if end_frame is None else end_frame
        end = (self.info["frame_pts"][end_frame] * Fraction(self.info["time_base"])
               if self.info["frame_pts"] and end_frame < self.info["frame_count"] else None)
        requests = []
        for request in self.plan["windows"]:
            window = request["window"]
            window_start = (Fraction(request["source_timeline"]["window_start_seconds"])
                            if request["source_timeline"] else Fraction(window["scene_first_frame"], 1)/self.fps + Fraction(window["model_first_frame"], 24))
            if (not request.get("inactive") and window_start + Fraction(window["valid_model_frames"], 24) > start
                    and (end is None or window_start < end)):
                requests.append(request)
        is_cancelled = lambda: self.stop.is_set() or cancelled()

        def prepare():
            for request in requests:
                if is_cancelled():
                    raise KeyboardInterrupt
                yield self._prepare_window(request, is_cancelled)

        self.prepared_stream = BoundedStream(prepare, capacity=1, name="mosaic-input-preparation", cancelled=is_cancelled)

        def generate():
            for prepared in self.prepared_stream:
                yield self._generate_prepared(prepared, is_cancelled)

        self.generated_stream = BoundedStream(generate, capacity=1, name="mosaic-h3-generator", cancelled=is_cancelled)
        self.prepared_stream.__enter__()
        self.generated_stream.__enter__()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        for stream in (self.generated_stream, self.prepared_stream):
            if stream is not None:
                stream.close()
        events = self.stage_events
        generated = [(e["start"],e["end"]) for e in events if e["stage"] == "generate"]
        overlaps = {stage: interval_overlap_seconds(generated,
                    [(e["start"],e["end"]) for e in events if e["stage"] == stage])
                    for stage in ("prepare", "compose")}
        self.pipeline_stats = {"enabled": self.prefetch,
            "queues": {name: stream.stats for name, stream in (("prepared",self.prepared_stream),("generated",self.generated_stream)) if stream is not None},
            "overlap_with_generation_seconds": overlaps, "events": events,
            "gpu_generators": 1, "media_rendered_or_played": False}
        atomic_json(self.job_dir / "processing-pipeline.json", self.pipeline_stats)

    def _window(self, request, cancelled):
        import numpy as np
        index = request["window"]["index"]
        if index in self.cache:
            return self.cache[index]
        if self.generated_stream is not None:
            while index not in self.generated_paths:
                try:
                    ready_index, path = next(self.generated_stream)
                except StopIteration as exc:
                    raise ValueError(f"H3窓 {index} を生成できませんでした") from exc
                self.generated_paths[ready_index] = path
            result = self.generated_paths[index]
        else:
            _, result = self._generate_prepared(self._prepare_window(request, cancelled), cancelled)
        canvas = request["canvas"]
        size = canvas["width"] * canvas["height"] * 3
        with rgb_reader(result) as reader:
            frames = [np.frombuffer(read_exact(reader, size), dtype=np.uint8).reshape(
                canvas["height"], canvas["width"], 3).copy()
                for _ in range(request["window"]["valid_model_frames"])]
        self.cache[index] = frames
        return frames

    def apply_frame(self, frame, frame_id, active, cancelled):
        if not active:
            return frame
        regions = self.schedule.at(frame_id) if self.schedule else (self.roi,)
        if not regions:
            return frame
        candidates = []
        instant = (self.info["frame_pts"][frame_id] * Fraction(self.info["time_base"])
                   if self.info["frame_pts"] else Fraction(frame_id,1)/self.fps)
        scene_first = self.scene_starts[bisect_right(self.scene_starts, frame_id) - 1]
        scene = self.scenes[scene_first]
        starts = self.window_starts[scene_first]
        current = bisect_right(starts, instant) - 1
        # Only windows whose actual time support can include this source PTS.
        first_candidate = bisect_right(starts, instant - self.plan["profile"]["context_seconds"])
        for request in scene[first_candidate:current + 1]:
            if request.get("inactive"):
                continue
            window = request["window"]
            if not window["scene_first_frame"] <= frame_id < window["scene_end_frame_exclusive"]:
                continue
            if self.info.get("frame_pts"):
                instant = self.info["frame_pts"][frame_id] * Fraction(self.info["time_base"])
                position = (instant - Fraction(request["source_timeline"]["window_start_seconds"])) * 24
            else:
                position = Fraction(frame_id - window["scene_first_frame"], 1) / self.fps * 24 - window["model_first_frame"]
            if 0 <= position < window["valid_model_frames"]:
                nearest = min(window["valid_model_frames"] - 1, int(position + Fraction(1, 2)))
                # Positive triangular edge weights; every valid source has support.
                weight = min(24.0, float(position) + 1, window["valid_model_frames"] - float(position))
                candidates.append((request, nearest, weight))
        if not candidates:
            raise ValueError(f"原frame {frame_id}を所有するH3窓がありません")
        live = {request["window"]["index"] for request, _, _ in candidates}
        self.cache = {key: value for key, value in self.cache.items() if key in live}
        roi = union(regions)
        ready = [(request, index, weight, self._window(request, cancelled)[index]) for request,index,weight in candidates]
        compose_start = time.monotonic()
        from .shape_masks import source_alpha
        from dataclasses import asdict
        entries = self.schedule.entries_at(frame_id) if self.schedule else [{"roi": asdict(self.roi)}]
        alpha = source_alpha(entries, roi, self.feather)
        windows = [{"pixels": generated, "canvas": request["canvas"],
                    "crop": request.get("source_frame_crops", {}).get(str(frame_id), request["crop"]),
                    "weight": weight} for request, index, weight, generated in ready]
        output = compose_frame(frame, self.info["width"], self.info["height"], roi, alpha, windows)
        if self.generation_busy:
            self.stage_events.append({"stage": "compose", "frame": frame_id, "start": compose_start, "end": time.monotonic()})
        return output
