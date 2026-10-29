"""Explicit release build. No Rust download, model access or ComfyUI changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mosaic_refiner.native_core import ABI_VERSION, LIBRARY_NAME, native_source_fingerprint


def check_running(destination):
    import psutil
    for process in psutil.process_iter(['name','exe']):
        if process.pid==os.getpid():continue
        try:
            exe=process.info['exe']
            if exe and Path(exe).parent==destination:
                raise ValueError("アプリの処理・終了を待ってから構築してください。実行中のアプリは自動停止しません")
            if 'python' in (process.info['name'] or '').lower():
                if any(Path(row.path).parent==destination for row in process.memory_maps()):
                    raise ValueError("Rust DLLを利用中の処理があります。終了後に構築してください")
        except (psutil.Error,OSError):continue


def find_cargo(explicit=None):
    """Use the caller's toolchain without assuming a developer's drive layout."""
    if explicit is not None:
        candidates = [Path(explicit).expanduser()]
    else:
        name = "cargo.exe" if os.name == "nt" else "cargo"
        found = shutil.which("cargo")
        home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
        candidates = ([Path(found)] if found else []) + [home / "bin" / name]
    for candidate in candidates:
        if candidate.is_file():
            # rustup dispatches by executable name; resolving cargo's symlink
            # to rustup.exe would turn `cargo build` into `rustup build`.
            return candidate.absolute()
    raise ValueError("cargoがありません。Rust/Cargoを導入し、PATHまたはCARGO_HOMEを設定するか、--cargoで実行ファイルを指定してください")


def find_rustc(cargo, environment):
    name = "rustc.exe" if os.name == "nt" else "rustc"
    found = shutil.which("rustc", path=environment.get("PATH", ""))
    candidates = ([Path(found)] if found else []) + [Path(cargo).with_name(name)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    raise ValueError("rustcがありません。Rust toolchainのPATH・CARGO_HOME・RUSTUP_HOMEを確認してください")


def main():
    os.environ["MOSAIC_REFINER_HOST"] = "python"  # Build records must not load a stale DLL.
    from mosaic_refiner.checkpoint import atomic_json
    parser = argparse.ArgumentParser(description="Rustの合成・数値検査coreを明示的に構築")
    parser.add_argument("--cargo", type=Path)
    args = parser.parse_args()
    destination = ROOT / "mosaic_refiner/_native"
    check_running(destination)
    try:
        cargo = find_cargo(args.cargo)
    except ValueError as exc:
        parser.error(str(exc))
    env = dict(os.environ)
    env["PATH"] = str(Path(cargo).parent) + os.pathsep + env.get("PATH", "")
    try:
        rustc = find_rustc(cargo, env)
        rustc_version = subprocess.check_output([str(rustc), "--version"], env=env, text=True).strip()
        cargo_version = subprocess.check_output([str(cargo), "--version"], env=env, text=True).strip()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error("Rust toolchainの版を確認できません。成果物は変更していません: " + str(exc))
    target = ROOT / "work/rust-target"
    subprocess.run([str(cargo), "generate-lockfile", "--manifest-path", str(ROOT / "native/host/Cargo.toml")],
                   env=env, cwd=ROOT, check=True) if not (ROOT / "native/host/Cargo.lock").exists() else None
    source_hash = native_source_fingerprint()
    env["MOSAIC_NATIVE_SOURCE_SHA"] = source_hash
    subprocess.run([str(cargo), "build", "--manifest-path", str(ROOT / "native/Cargo.toml"),
                    "--release", "--locked", "--target-dir", str(target)], env=env, cwd=ROOT, check=True)
    subprocess.run([str(cargo), "build", "--manifest-path", str(ROOT / "native/host/Cargo.toml"),
                    "--release", "--locked", "--target-dir", str(target)], env=env, cwd=ROOT, check=True)
    if native_source_fingerprint() != source_hash:
        raise ValueError("構築中にRust sourceが変更されました。成果物を確定せず停止します")
    artifact = target / "release" / LIBRARY_NAME
    check_running(destination)
    destination.mkdir(exist_ok=True)
    library = destination / LIBRARY_NAME
    partial = library.with_suffix(library.suffix + ".partial")
    shutil.copyfile(artifact, partial)
    with partial.open("rb+") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, library)
    record = {"abi_version": ABI_VERSION, "source_sha256": source_hash,
              "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "rustc": rustc_version,
              "cargo": cargo_version,
              "profile": "release", "library_bytes": library.stat().st_size}
    atomic_json(destination / "build.json", record)
    from mosaic_refiner.native_host import HOST_NAME
    host = destination / HOST_NAME
    shutil.copyfile(target / "release" / HOST_NAME, host.with_suffix(".partial"))
    os.replace(host.with_suffix(".partial"), host)
    atomic_json(destination / "host-build.json", {"source_sha256": source_hash,
        "host_sha256": hashlib.sha256(host.read_bytes()).hexdigest(), "host_bytes": host.stat().st_size,
        "ui_sha256": hashlib.sha256((ROOT / "mosaic_refiner/web.html").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "rustc": record["rustc"], "profile": "release"})
    from mosaic_refiner.native_jobs import LIBRARY_NAME as JOBS_LIBRARY_NAME
    jobs_library = destination / JOBS_LIBRARY_NAME
    shutil.copyfile(target / "release" / JOBS_LIBRARY_NAME, jobs_library.with_suffix(".partial"))
    os.replace(jobs_library.with_suffix(".partial"), jobs_library)
    atomic_json(destination / "jobs-build.json", {"source_sha256": source_hash,
        "library_sha256": hashlib.sha256(jobs_library.read_bytes()).hexdigest(), "abi_version": 1,
        "library_bytes": jobs_library.stat().st_size})
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
