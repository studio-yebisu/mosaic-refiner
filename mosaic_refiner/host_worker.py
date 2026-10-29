"""One-shot processing worker. Scheduling, session, HTTP and cancellation belong to Rust.

Only numeric metadata/events go to stdout. No media is displayed; no import starts
downloads, a server or inference. The existing app/GPU environments remain separate.
"""
import contextlib
import json
from pathlib import Path
import sys
import threading


def emit(kind, data):
    sys.__stdout__.write(json.dumps({"kind": kind, "data": data}, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def metadata():
    from .processing_options import catalog
    from .h3_lora_profiles import available_profiles
    from .learned_detection import model_path
    from .restoration_models import ready
    return {"processing_presets": catalog(), "h3_lora_profiles": available_profiles(),
            "detector_ready": model_path().is_file(), "restoration_ready": ready(),
            "restoration_profiles": {name:ready(name) for name in available_profiles()}}


def execute(spec):
    from .application import job_from_spec
    from .checkpoint import atomic_json
    job = job_from_spec(spec, lambda **event: emit("progress", event))
    stop = threading.Event()
    marker = Path(str(spec["job_dir"])+".host-cancel.json")

    def cancellation():
        while not stop.wait(.05):
            if marker.exists():
                job.cancel()
                return

    def progress(**event):
        if spec.get("keep_intermediates") and (job.job_dir / "manifest.json").is_file():
            keep = job.job_dir / "keep-intermediates.json"
            if not keep.exists():
                atomic_json(keep, {"keep": True})
        emit("progress", event)

    job.progress = progress
    thread = threading.Thread(target=cancellation, daemon=True)
    thread.start()
    try:
        return job.execute()
    finally:
        stop.set()
        thread.join()


def request(operation, data):
    from .runtime_settings import use_local_tools
    use_local_tools()
    if operation == "metadata":
        return metadata()
    if operation == "probe":
        from .media import command
        result = json.loads(command(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration", "-of", "json", data["source"]]))
        if not result.get("streams"):
            raise ValueError("映像streamを読み取れません")
        return {**result["streams"][0], "duration": result.get("format", {}).get("duration")}
    if operation == "options":
        from .processing_options import resolve_options
        options=data.get("processing_options")
        if options is None:
            from .setup_environment import installed_lora_profile
            options={"lora_profile":installed_lora_profile()}
        return resolve_options(options).to_dict()
    if operation == "execute":
        return execute(data)
    if operation == "cleanup":
        from scripts.completed_job_cleanup import cleanup_job
        return cleanup_job(Path(data["job_dir"]), manual=data.get("manual") is True)
    if operation == "preview":
        from .media import command
        target = Path(data["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(command(["ffmpeg", "-v", "error", "-nostdin", "-noautorotate",
                "-i", data["source"], "-map", "0:v:0", "-frames:v", "1", "-vf",
                "scale=640:360:force_original_aspect_ratio=decrease", "-f", "image2pipe",
                "-vcodec", "png", "pipe:1"]))
        return {"preview": str(target)}
    if operation in ("prepare-detector", "prepare-restoration"):
        if operation == "prepare-detector":
            from .learned_detection import setup
        else:
            from .restoration_models import setup
        kwargs = {"lora_profile": data.get("lora_profile", "unlocked_v2"),
                  "accept_terms": data.get("accept_terms") is True} if operation == "prepare-restoration" else {}
        setup(progress=lambda **event: emit("progress", event), **kwargs)
        return metadata()
    if operation == "cli":
        from .__main__ import main
        # Legacy processing operations keep their exact numeric behavior. CLI
        # parsing/help/status/cancel and process supervision are native.
        import io
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(data["argv"])
        value = output.getvalue()
        try:
            result = json.loads(value)
        except ValueError:
            result = {"text": value}
        return {"exit_code": code, "result": result}
    raise ValueError("未知のworker operation")


def main():
    try:
        data = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            result = request(data["operation"], data.get("data", {}))
        emit("result", result)
        return 0
    except BaseException as exc:
        # Never report a killed/crashed worker as complete.
        emit("error", {"error": str(exc) or type(exc).__name__})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
