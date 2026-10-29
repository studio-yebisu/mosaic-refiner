"""Sequential CPU diagnostic pipeline. No model/server interaction."""

from dataclasses import asdict, replace
from fractions import Fraction
import os
from pathlib import Path
import shutil
import tempfile

from .checkpoint import (PIPELINE_VERSION, SCHEMA_VERSION, atomic_json, digest_config,
                         digest_file, implementation_fingerprint, job_lock, load_json, summarize)
from .geometry import DiagnosticCompositor, Rect
from .media import (ChunkEncoder, MediaError, encode_output, probe, read_exact, rgb_reader,
                    tool_version, verify_output)
from .planning import plan_cfr, timeline_units


def _check_space(directory: Path, needed: int) -> None:
    if shutil.disk_usage(directory).free < needed + 1024 * 1024:
        raise MediaError("空きdiskが次の保存に必要な保守的見積もりを下回りました")


def run(
    source: Path,
    output: Path,
    job_dir: Path,
    roi: Rect,
    backend: str = "identity",
    feather: int = 0,
    unit_seconds: Fraction = Fraction(15),
    context_seconds: int = 5,
    first_frame: int = 0,
    end_frame: int | None = None,
    stop_after_units: int | None = None,
    processor=None,
    progress=None,
    allow_timeline=False,
    source_info=None,
) -> dict:
    source, output, job_dir = source.resolve(), output.resolve(), job_dir.resolve()
    if source == output:
        raise ValueError("入力原本を出力先にできません")
    if output.suffix.lower() not in (".nut", ".mp4"):
        raise ValueError("CPU診断版の出力は.nut（FFV1）または.mp4（lossless RGB）です")
    if source == job_dir or job_dir in source.parents or job_dir in output.parents:
        raise ValueError("入力・最終出力はjobディレクトリの外へ置いてください")
    if stop_after_units is not None and stop_after_units < 1:
        raise ValueError("stop-after-unitsは1以上です")
    if not output.exists():
        output.parent.mkdir(parents=True,exist_ok=True)
        _check_space(output.parent,0)
        handle,name=tempfile.mkstemp(prefix='.mosaic-write-check-',dir=output.parent)
        os.close(handle)
        Path(name).unlink()
    info = source_info if source_info is not None else probe(source, allow_timeline=allow_timeline or processor is not None)
    end_frame = info.frame_count if end_frame is None else end_frame
    if not 0 <= first_frame < end_frame <= info.frame_count:
        raise ValueError("対象frame区間は0 <= first < end <= frame数です")
    compositor = processor or DiagnosticCompositor(info.width, info.height, roi, backend, feather)
    plan = plan_cfr(info.frame_count, Fraction(info.fps), unit_seconds, context_seconds)
    if info.frame_pts:
        cuts = tuple(sorted({r["window"]["scene_first_frame"] for r in processor.plan["windows"]
                             if r["window"]["scene_first_frame"] > 0})) if processor else ()
        plan = replace(plan, units=timeline_units(info.frame_pts, info.time_base, unit_seconds, cuts))
    config = {
        "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
        "implementation": implementation_fingerprint(),
        "source_sha256": digest_file(source), "source_info": info.to_dict(),
        "backend": backend, "roi": asdict(roi), "feather": feather,
        "first_frame": first_frame, "end_frame": end_frame,
        "unit_seconds": str(unit_seconds), "context_seconds": context_seconds,
        "ffmpeg": tool_version("ffmpeg"), "ffprobe": tool_version("ffprobe"),
    }
    from .native_core import identity as core_identity
    config["cpu_core"] = core_identity()
    if processor is not None:
        config["processor"] = processor.fingerprint()
    job_id = digest_config(config)
    manifest_path = job_dir / "manifest.json"
    with job_lock(job_dir):
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            if manifest.get("job_id") != job_id or manifest.get("config") != config:
                raise ValueError("入力/設定/backend版がjobと異なります。新しい--job-dirを指定してください")
        else:
            if any(p.name != ".lock" for p in job_dir.iterdir()):
                raise ValueError("manifestのない既存ファイルがjob内にあります。新しい--job-dirを指定してください")
            manifest = {
                "schema_version": SCHEMA_VERSION, "job_id": job_id, "config": config,
                "source": str(source), "state": "pending", "units": [
                    {**asdict(unit), "state": "pending"} for unit in plan.units],
            }
            atomic_json(manifest_path, manifest)
        if len(manifest.get("units", [])) != len(plan.units):
            raise ValueError("manifestの区間数が計画と異なります")
        for expected, actual in zip(plan.units, manifest["units"]):
            if any(actual.get(key) != value for key, value in asdict(expected).items()):
                raise ValueError("manifestのframe割当が計画と異なります")

        # An existing result is reusable only if it is our recorded, intact output.
        if output.exists():
            if (manifest.get("state") == "complete" and manifest.get("output") == str(output)
                    and manifest.get("output_sha256") == digest_file(output)):
                result = summarize(job_dir)
                result["output_integrity_checked"] = True
                return result
            raise ValueError("出力先が既に存在します。別のファイル名を指定してください")
        output.parent.mkdir(parents=True, exist_ok=True)
        cancel_path = job_dir / "cancel.json"
        # A fresh invocation explicitly resumes a cancelled job.
        cancel_path.unlink(missing_ok=True)
        manifest.update(state="running", last_error=None)
        manifest["stage"] = ("h3-region-video" if getattr(processor, "ai_backend", False) else
                             "synthetic-window-no-ai" if processor else "cpu-diagnostic-no-ai")
        atomic_json(manifest_path, manifest)
        reused = 0
        generated = 0
        temporary_output = None
        try:
            # All complete artifacts are rehashed before any is reused.
            for unit in manifest["units"]:
                path = job_dir / f"chunk-{unit['index']:06d}.nut"
                if unit["state"] == "complete" and path.is_file() and unit.get("sha256") == digest_file(path):
                    reused += 1
                else:
                    unit["state"] = "pending"
                    unit.pop("sha256", None)
            atomic_json(manifest_path, manifest)
            pending = [unit for unit in manifest["units"] if unit["state"] != "complete"]
            if pending:
                # One sequential decode per invocation, including intact units
                # between holes. On resume FFmpeg decodes/discards the prefix.
                start = pending[0]["first_frame"]
                size = info.width * info.height * 3
                if processor is not None and hasattr(processor, "start"):
                    processor.start(max(start, first_frame), cancel_path.exists, end_frame)
                with rgb_reader(source, start, info) as reader:
                    for unit in manifest["units"]:
                        if unit["end_frame_exclusive"] <= start:
                            continue
                        count = unit["end_frame_exclusive"] - unit["first_frame"]
                        if unit["state"] == "complete":
                            for _ in range(count):
                                if cancel_path.exists():
                                    raise KeyboardInterrupt
                                read_exact(reader, size)
                            continue
                        _check_space(job_dir, count * info.width * info.height * 4)
                        chunk_path = job_dir / f"chunk-{unit['index']:06d}.nut"
                        partial = chunk_path.with_suffix(".nut.partial")
                        unit["state"] = "running"
                        atomic_json(manifest_path, manifest)
                        encoder = ChunkEncoder(partial, info)
                        try:
                            for frame_id in range(unit["first_frame"], unit["end_frame_exclusive"]):
                                if cancel_path.exists():
                                    raise KeyboardInterrupt
                                frame = read_exact(reader, size)
                                active = first_frame <= frame_id < end_frame
                                encoder.write(processor.apply_frame(frame, frame_id, active, cancel_path.exists)
                                              if processor else compositor.apply(frame, active))
                                if progress is not None:
                                    progress(phase="領域合成・次のH3窓を生成" if getattr(processor,"generation_busy",False) else "領域合成",
                                             frame=frame_id + 1, total=info.frame_count)
                        except BaseException:
                            encoder.close(abort=True)
                            raise
                        else:
                            encoder.close()
                        verified = probe(partial)
                        if verified.frame_count != count or verified.width != info.width or verified.height != info.height:
                            raise MediaError("checkpointのframe数/寸法が一致しません")
                        with partial.open("rb+") as stream:
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(partial, chunk_path)
                        unit.update(state="complete", sha256=digest_file(chunk_path), bytes=chunk_path.stat().st_size)
                        atomic_json(manifest_path, manifest)
                        generated += 1
                        if stop_after_units is not None and generated >= stop_after_units:
                            raise KeyboardInterrupt
            if cancel_path.exists():
                raise KeyboardInterrupt
            if processor is not None and hasattr(processor, "close"):
                processor.close()
                manifest["processing_pipeline"] = processor.pipeline_stats
            chunks = [job_dir / f"chunk-{unit['index']:06d}.nut" for unit in manifest["units"]]
            _check_space(output.parent, sum(path.stat().st_size for path in chunks) * 2)
            manifest["state"] = "encoding"
            atomic_json(manifest_path, manifest)
            if progress is not None:
                progress(phase="元の時刻・音声で保存")
            # Reserve a new same-volume path exclusively. A deterministic name
            # can collide with the input or an unrelated existing file, which
            # FFmpeg -y and the failure cleanup must never touch.
            handle, name = tempfile.mkstemp(prefix=f".{output.stem}.{job_id[:12]}.",
                                            suffix=f".partial{output.suffix}", dir=output.parent)
            temporary_output = Path(name)
            os.close(handle)
            encode_output(chunks, info, source, temporary_output, cancel_path.exists)
            if cancel_path.exists():
                raise KeyboardInterrupt
            if progress is not None:
                progress(phase="完成動画を検査")
            verification = verify_output(source, temporary_output, info)
            if cancel_path.exists():
                raise KeyboardInterrupt
            verification["ai_backend_executed"] = bool(getattr(processor, "ai_backend", False))
            # Model execution does not prove the mosaic disappeared.
            verification["ai_restoration"] = False
            verification["restoration_pipeline_executed"] = bool(getattr(processor, "ai_backend", False))
            verification["restoration_effect_proven"] = False
            verification["quality_reviewed"] = False
            if digest_file(source) != config["source_sha256"]:
                raise MediaError("処理中に入力原本が変更されました。新jobで再実行してください")
            if implementation_fingerprint() != config["implementation"]:
                raise MediaError("処理中に実装が変更されました。新jobで再実行してください")
            with temporary_output.open("rb+") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            if cancel_path.exists():
                raise KeyboardInterrupt
            # Publish without replacing a user file created while encoding.
            # Windows rename and POSIX hard link refuse an existing pathname.
            try:
                if os.name=='nt':os.rename(temporary_output,output)
                else:os.link(temporary_output, output)
            except FileExistsError as exc:
                raise MediaError("処理中に出力先が作成されました。上書きせず停止しました") from exc
            manifest.update(state="complete", output=str(output), output_sha256=digest_file(output),
                            verification=verification, reused_units=reused, generated_units=generated)
            atomic_json(manifest_path, manifest)
        except KeyboardInterrupt:
            manifest["state"] = "cancelled"
            for unit in manifest["units"]:
                if unit["state"] == "running":
                    unit["state"] = "pending"
            atomic_json(manifest_path, manifest)
        except Exception as exc:
            manifest.update(state="failed", last_error=str(exc))
            atomic_json(manifest_path, manifest)
            raise
        finally:
            if processor is not None and hasattr(processor, "close"):
                processor.close()
            # Only the path exclusively reserved by this invocation is owned.
            if temporary_output is not None:
                try:
                    temporary_output.unlink(missing_ok=True)
                except OSError:
                    # Antivirus/OS locks may leave our uniquely named staging
                    # file. Cleanup must not hide the processing error or turn
                    # a validated, recorded output into a failed job.
                    pass
        result = summarize(job_dir)
        result.update(integrity_checked=True, reused_units=reused, generated_units=generated)
        return result
