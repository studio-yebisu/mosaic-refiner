"""Verified terminal-job maintenance, separate from the running inference code."""
import argparse
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaic_refiner.checkpoint import atomic_json, digest_file, job_lock, load_json

MEDIA = {"low-reference.nut", "high-reference.nut", "low-write-mask.pgm", "high-write-mask.pgm",
         "low-mask.nut", "high-mask.nut", "restored-low-reference.nut", "restored-high-reference.nut"}
VERIFIED = ("pts_exact", "tail_duration_exact", "audio_packets_exact", "full_decode")


def contained_file(file, root):
    root = root.resolve()
    if file.is_symlink() or (hasattr(file, "is_junction") and file.is_junction()):
        return False
    resolved = file.resolve()
    if root not in resolved.parents:
        return False
    current = file.parent
    while current != root and root in current.resolve().parents:
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            return False
        current = current.parent
    return file.is_file()


def candidates(job, manifest, comfy_output):
    for unit in manifest.get("units", []):
        index = unit.get("index")
        if type(index) is int and index >= 0 and unit.get("state") == "complete":
            yield job / f"chunk-{index:06d}.nut", unit.get("sha256"), job
    windows = job / "windows"
    if not windows.is_dir() or windows.is_symlink() or (hasattr(windows, "is_junction") and windows.is_junction()):
        return
    for window in windows.iterdir():
        if not re.fullmatch(r"\d{6,}(?:-incomplete-\d+)?", window.name):
            continue
        if window.is_symlink() or (hasattr(window, "is_junction") and window.is_junction()):
            continue
        receipt = window / "input/manifest.json"
        if contained_file(receipt, job):
            data = load_json(receipt)
            for name, record in data.get("artifacts", {}).items():
                if name in MEDIA:
                    yield window / "input" / name, record.get("sha256"), job
        receipt = window / "input/restoration.json"
        if contained_file(receipt, job):
            data = load_json(receipt)
            if data.get("state") == "complete":
                for name, record in data.get("artifacts", {}).items():
                    if name in MEDIA:
                        yield window / "input" / name, record.get("sha256"), job
        receipt = window / "generation/result.json"
        if not contained_file(receipt, job):
            continue
        result = load_json(receipt)
        if result.get("state") != "complete":
            continue
        sha = result.get("sha256")
        yield window / "generation/generated.mp4", sha, job
        history, workflow = window / "generation/history.json", window / "generation/workflow.json"
        if not contained_file(history, job) or not contained_file(workflow, job):
            continue
        graph = load_json(workflow)
        prefix = graph.get("save", {}).get("inputs", {}).get("filename_prefix", "")
        if not re.fullmatch(r"mosaic-refiner/[a-f0-9]{32}", prefix):
            continue
        data = load_json(history)
        if not data.get("status", {}).get("completed"):
            continue
        token = prefix.split("/")[1]
        for entry in data.get("outputs", {}).get("save", {}).get("images", []):
            name = entry.get("filename", "")
            # Shared ComfyUI output: exact own prefix, own receipt and same bytes.
            if (entry.get("type", "output") == "output" and entry.get("subfolder") == "mosaic-refiner"
                    and re.fullmatch(re.escape(token) + r"_[A-Za-z0-9_.-]+\.mp4", name)):
                yield comfy_output / "mosaic-refiner" / name, sha, comfy_output


def cleanup_job(job, comfy_output=None, *, manual=False):
    if comfy_output is None:
        from mosaic_refiner.runtime_settings import RUNTIME
        comfy_output = RUNTIME.root / 'output'
    job = Path(job)
    if job.is_symlink() or (hasattr(job, "is_junction") and job.is_junction()):
        return {"state": "skipped", "reason": "linked-directory"}
    job = job.resolve()
    manifest_path = job / "manifest.json"
    if not manifest_path.is_file():
        return {"state": "skipped", "reason": "no-manifest"}
    with job_lock(job):
        manifest = load_json(manifest_path)
        if manifest.get("state") != "complete":
            return {"state": "skipped", "reason": "not-complete"}
        if not manual and (job / "keep-intermediates.json").exists():
            return {"state": "retained", "reason": "keep-intermediates"}
        report_path = job / "cleanup.json"
        previous = load_json(report_path) if report_path.exists() else {}
        if previous.get("state") == "complete":
            return previous
        verification = manifest.get("verification", {})
        if not all(verification.get(key) is True for key in VERIFIED):
            return {"state": "skipped", "reason": "not-verified"}
        source, output = Path(manifest["source"]), Path(manifest["output"])
        if (not output.is_file() or output.resolve() == source.resolve()
                or job == output.resolve() or job in output.resolve().parents):
            return {"state": "skipped", "reason": "invalid-output"}
        if digest_file(output) != manifest.get("output_sha256"):
            return {"state": "skipped", "reason": "output-hash-mismatch"}
        report = {"state": "running", "policy": "verified-completion-only-v1", "started_at": time.time(),
                  "output_sha256": manifest["output_sha256"], "removed": previous.get("removed", []),
                  "skipped": [], "bytes_freed": previous.get("bytes_freed", 0)}
        atomic_json(report_path, report)
        try:
            for file, expected, root in candidates(job, manifest, Path(comfy_output)):
                if not file.exists():
                    continue
                label = ("job/" if root == job else "comfy-output/") + file.relative_to(root).as_posix()
                if (not contained_file(file, root) or file.resolve() in (source.resolve(), output.resolve())
                        or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)):
                    report["skipped"].append({"file": label, "reason": "untrusted-path-or-receipt"})
                    continue
                if digest_file(file) != expected:
                    report["skipped"].append({"file": label, "reason": "changed-bytes"})
                    continue
                size = file.stat().st_size
                file.unlink()
                report["removed"].append({"file": label, "bytes": size, "sha256": expected})
                report["bytes_freed"] += size
                atomic_json(report_path, report)
            report.update(state="complete", completed_at=time.time())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report.update(state="blocked", error=str(exc))
        atomic_json(report_path, report)
        return report


def scan(roots, since):
    for root in roots:
        if not root.is_dir() or root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
            continue
        for directory in root.iterdir():
            manifest = directory / "manifest.json"
            if not directory.is_dir() or not manifest.is_file() or manifest.stat().st_mtime < since:
                continue
            try:
                # cleanup_job takes the writer lock before reading. A pre-read
                # here can hold a large manifest open during Windows replace.
                result = cleanup_job(directory)
                if result["state"] == "complete" and not result.get("announced"):
                    print(f"中間ファイル整理完了: job={directory.name}, {result['bytes_freed']:,} bytes", flush=True)
                    result["announced"] = True
                    atomic_json(directory / "cleanup.json", result)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"整理は保留: job={directory.name}, {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="完成検査済みjobの記録済み中間動画だけを整理")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--keep-job", type=Path, help="指定jobの中間ファイルを再調整用に保持")
    parser.add_argument("--allow-job", type=Path, help="指定jobの保持指定を解除")
    args = parser.parse_args()
    if args.keep_job:
        atomic_json(args.keep_job / "keep-intermediates.json", {"keep": True})
    elif args.allow_job:
        (args.allow_job / "keep-intermediates.json").unlink(missing_ok=True)
    elif args.job_dir:
        print(cleanup_job(args.job_dir, manual=True))
    elif args.watch:
        parser.error("自動整理は廃止しました。履歴画面、または --job-dir で完成済みjobを明示してください")
    else:
        parser.error("--watch / --job-dir / --keep-job / --allow-jobを指定してください")


if __name__ == "__main__":
    main()
