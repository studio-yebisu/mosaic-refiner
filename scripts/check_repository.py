"""Check bootstrap structure and tracked-file exclusions without scanning media."""

from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "AGENTS.md", "README.md", ".gitignore", "pyproject.toml",
    "docs/requirements.md", "docs/design.md", "docs/implementation-plan.md", "docs/ai-setup.md",
    "mosaic_refiner/__init__.py", "mosaic_refiner/__main__.py",
    "mosaic_refiner/planning.py", "LICENSE", "NOTICE.md", "THIRD_PARTY_NOTICES.md",
    "setup-app.cmd", "setup-app.ps1", "mosaic_refiner/runtime_settings.py",
    "mosaic_refiner/setup_environment.py", "mosaic_refiner/setup_assets/recipe.json",
    "mosaic_refiner/setup_assets/NOTICE.md", "docs/hardware-requirements.md",
)
PRIVATE_DIRECTORIES = {"models", "inputs", "outputs", "jobs", "cache", "logs", "secrets", "credentials"}
PRIVATE_SUFFIXES = {
    ".safetensors", ".ckpt", ".pth", ".pt", ".onnx", ".gguf", ".mp4", ".mkv",
    ".mov", ".avi", ".webm", ".nut", ".rgb", ".pgm", ".wav", ".mp3", ".flac", ".sqlite", ".sqlite3", ".db", ".log",
}


def main() -> int:
    errors = [f"必要ファイルがありません: {name}" for name in REQUIRED if not (ROOT / name).is_file()]
    try:
        with (ROOT / "pyproject.toml").open("rb") as stream:
            config = tomllib.load(stream)
        if config["project"]["name"] != "mosaic-refiner":
            errors.append("プロジェクト名が一致しません")
    except (OSError, tomllib.TOMLDecodeError, KeyError) as exc:
        errors.append(f"pyproject.toml を確認できません: {exc}")
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=False,
    )
    if tracked.returncode:
        errors.append("Git の tracked file を取得できません")
    else:
        for name in tracked.stdout.decode("utf-8").split("\0"):
            if not name:
                continue
            path = Path(name)
            if path.parts[0] in PRIVATE_DIRECTORIES or path.suffix.lower() in PRIVATE_SUFFIXES:
                errors.append(f"private 素材等が Git 対象です: {name}")
            if path.name.startswith(".env") and path.name != ".env.example":
                errors.append(f"ローカル設定が Git 対象です: {name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("基本構成と Git 対象ファイルの検査に合格しました（GPU・画質は未検証）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
