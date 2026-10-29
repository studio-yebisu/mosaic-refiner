"""Shared GUI/CLI application job API with explicit analysis and AI states."""
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import threading
import time
import uuid

from .checkpoint import atomic_json, digest_file, load_json, request_cancel, implementation_fingerprint
from .detection import RegionSchedule, analyze, union
from .geometry import Rect
from .h3_processing import H3Processor
from .media import probe
from .pipeline import run
from .processing_options import resolve_options


class ApplicationJob:
    def __init__(self, source, output, job_dir, roi=None, backend="h3", progress=lambda **kw: None,
                 context_seconds=5, unit_seconds=15, processing_options=None):
        self.source, self.output, self.job_dir = map(Path, (source, output, job_dir))
        self.roi, self.backend, self.progress = roi, backend, progress
        self.cancel_event = threading.Event()
        self.context_seconds, self.unit_seconds = context_seconds, Fraction(unit_seconds)
        self.processing_options = resolve_options(processing_options)

    def cancel(self):
        self.cancel_event.set()
        if (self.job_dir / "manifest.json").is_file():
            request_cancel(self.job_dir)

    def execute(self):
        started = time.monotonic()
        times = {}
        self.cancel_event.clear()
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.progress(phase="入力の時刻・音声を確認")
        info = probe(self.source, allow_timeline=True)
        times["source_index_seconds"] = time.monotonic()-started
        if self.cancel_event.is_set():
            return {"state": "cancelled", "stage": "analysis"}
        schedule = None
        analysis_start = time.monotonic()
        if self.roi is None:
            from .learned_detection import identity as detector_identity
            detector = detector_identity()
            report = self.job_dir.with_name(self.job_dir.name + ".analysis.json")
            source_hash = digest_file(self.source)
            analysis_fingerprint = {"implementation": implementation_fingerprint(), "detector": detector}
            if report.exists():
                saved = load_json(report)
                if saved.get("source_sha256") != source_hash:
                    raise ValueError("調査後に入力が変わりました。新しいjobで処理してください")
                if saved.get("analysis_fingerprint") != analysis_fingerprint:
                    raise ValueError("検出実装/モデルが調査時と異なります。新しい処理を開始してください")
                schedule = RegionSchedule({int(k): v for k, v in saved["records"].items()},
                                          saved["scene_cuts"], saved["max_gap_frames"], saved.get("metadata"))
            else:
                try:
                    schedule = analyze(self.source, info, self.cancel_event.is_set, self.progress, detector)
                except KeyboardInterrupt:
                    return {"state": "cancelled", "stage": "analysis"}
                if implementation_fingerprint() != analysis_fingerprint["implementation"]:
                    raise ValueError("検出中に実装が変更されました。新しい処理を開始してください")
                atomic_json(report, {**schedule.to_dict(), "source_sha256": source_hash,
                                     "analysis_fingerprint": analysis_fingerprint})
            boxes = [Rect(**r["roi"]) for entries in schedule.records.values() for r in entries]
            if not boxes:
                raise ValueError("対象領域を検出できませんでした。原本は保全しています。設定を調整するか手動で対象を指定してください")
            roi = union(boxes)
        else:
            roi = self.roi
        times["analysis_seconds"] = time.monotonic()-analysis_start
        roi.validate(info.width, info.height)
        context = self.context_seconds
        if context == "auto":
            from .h3_runtime import api
            # Longer context avoids repeated conditioning/overlap for small crops.
            # This chooses a profile on the observed GPU; it is not a 12GB proof.
            memory = max((d.get("vram_total",0) for d in api("/system_stats").get("devices",[])),default=0) if self.backend == "h3" else 0
            context = 15 if memory >= 20*1024**3 else 5
        self.progress(phase="処理計画を準備", context_seconds=context,
                      detection_summary=schedule.metadata if schedule else {"method":"manual"})
        plan_start = time.monotonic()
        feather = min(8, roi.width // 2, roi.height // 2)
        processor = H3Processor(self.source, self.job_dir, roi, feather=feather,
                                context_seconds=context,
                                scene_cuts=schedule.scene_cuts if schedule else (),
                                progress=self.progress, schedule=schedule,source_info=info,
                                processing_options=self.processing_options) if self.backend == "h3" else None
        times["planning_seconds"] = time.monotonic()-plan_start
        if processor:
            canvases = sorted({(r["canvas"]["low_width"],r["canvas"]["low_height"],
                                r["canvas"]["width"],r["canvas"]["height"])
                               for r in processor.plan["windows"] if not r.get("inactive")})
            self.progress(generation_summary={"canvases": canvases, "shape_masks": bool(schedule and schedule.metadata.get("shape_masks")),
                "following_windows": sum(r.get("crop_method") == "scene-local-fixed-size-follow-v1" for r in processor.plan["windows"]),
                "active_windows": sum(not r.get("inactive") for r in processor.plan["windows"]),
                "bounded_pipeline": processor.prefetch, "h3_regeneration_required": True,
                "processing_options": self.processing_options.summary()})
        # pipeline removes an old cancel request on explicit resume. Reapply a
        # live controller cancellation at the first progress/safe point.
        def progress(**event):
            if self.cancel_event.is_set():
                request_cancel(self.job_dir)
            self.progress(**event)
        if processor:
            processor.progress = progress
        if self.cancel_event.is_set():
            return {"state": "cancelled", "stage": "analysis"}
        execute_start = time.monotonic()
        result = run(self.source, self.output, self.job_dir, roi, backend=self.backend,
                   feather=feather, processor=processor, progress=progress, allow_timeline=True,
                   context_seconds=context, unit_seconds=self.unit_seconds,source_info=info)
        times.update(processing_seconds=time.monotonic()-execute_start,total_seconds=time.monotonic()-started,
                     context_seconds=context,media_rendered_or_played=False,
                     processing_options=self.processing_options.summary())
        if processor:
            times["processing_pipeline"] = getattr(processor,"pipeline_stats",{"reused_complete_output":True})
        atomic_json(self.job_dir / "timings.json",times)
        result["timings"] = times
        return result


def new_job(source, directory):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    suffix = ".nut" if source.suffix.lower() == ".nut" else ".mp4"
    job_dir = directory / uuid.uuid4().hex
    output = source.with_name(source.stem + ".refined-" + job_dir.name + suffix)
    index = 1
    while output.exists():
        output = source.with_name(source.stem + f".refined-{job_dir.name}-{index}" + suffix)
        index += 1
    return {"source": str(source), "output": str(output), "job_dir": str(job_dir), "backend": "h3", "roi": None}


def save_session(spec, path):
    atomic_json(Path(path), spec)


def job_from_spec(spec, progress=lambda **kw: None):
    return ApplicationJob(spec["source"], spec["output"], spec["job_dir"],
                          Rect(**spec["roi"]) if spec.get("roi") else None,
                          spec.get("backend", "h3"), progress,
                          spec.get("context_seconds", 5), spec.get("unit_seconds", 15),
                          spec.get("processing_options"))
