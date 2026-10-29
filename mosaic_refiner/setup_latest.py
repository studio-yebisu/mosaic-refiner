"""Resolve explicitly requested code/tool releases. Import performs no I/O."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

from .model_setup import SetupError


ALLOWED_REPOSITORIES = frozenset({
    "Comfy-Org/ComfyUI",
    "seitanism/ComfyUI-H3-Motion-Context-MultiRef",
    "dntpi/ComfyUI-H3-Video-Upsampler",
    "Kosinkadink/ComfyUI-VideoHelperSuite",
})
CORE_REPOSITORY = "Comfy-Org/ComfyUI"
FFMPEG_REPOSITORY = "GyanD/codexffmpeg"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 300
SHA = re.compile(r"[0-9a-fA-F]{40}")
PIN = re.compile(r"(comfy[-_]kitchen|comfy[-_]aimdo)\s*==\s*([0-9][A-Za-z0-9.+!_-]*)", re.I)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Canonical API/raw/codeload endpoints require no redirects. Refuse
        # before issuing any request to an unreviewed destination.
        raise SetupError("最新構成の取得先が転送を要求しました。自動追従せず停止します")


def _official_open(request, timeout):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _cancelled(cancelled):
    if cancelled():
        raise KeyboardInterrupt


def _reference(value):
    if (not isinstance(value, str) or not value or len(value) > 200
            or any(ord(c) < 33 or ord(c) == 127 for c in value)):
        raise SetupError("配布元のtag / branchが不正です。旧版へ戻さず停止します")
    return value


def _stable_release(data):
    if not isinstance(data, dict) or data.get("draft") is not False or data.get("prerelease") is not False:
        raise SetupError("最新の安定releaseを確認できません。旧版へ戻さず停止します")
    return _reference(data.get("tag_name"))


def _size(response, maximum):
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    if not re.fullmatch(r"[0-9]+", str(value)):
        raise SetupError("配布元の取得容量が不正です")
    length = int(value)
    if length <= 0 or length > maximum:
        raise SetupError("配布元の取得容量が上限を超えています")
    return length


def _open(url, opener):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.hostname not in {"api.github.com", "codeload.github.com", "raw.githubusercontent.com"}):
        raise SetupError("公式のHTTPS取得先以外は使用しません")
    response = opener(Request(url, headers={"User-Agent": "MosaicRefiner-setup/0.1", "Accept": "application/vnd.github+json"}),
                      timeout=REQUEST_TIMEOUT)
    try:
        # These three endpoints need no cross-origin redirects. Check the final
        # URL as well, including when a caller injects a different HTTP opener.
        final = response.geturl() if hasattr(response, "geturl") else url
        if final != url or getattr(response, "status", 200) != 200:
            raise SetupError("公式取得先の転送先またはHTTP応答を確認できません")
        return response
    except BaseException:
        response.close()
        raise


def _read(url, opener, requests, cancelled):
    _cancelled(cancelled)
    with _open(url, opener) as response:
        expected = _size(response, MAX_METADATA_BYTES)
        data = response.read(MAX_METADATA_BYTES + 1)
    _cancelled(cancelled)
    if not data or len(data) > MAX_METADATA_BYTES or (expected is not None and len(data) != expected):
        raise SetupError("配布元のmetadata / requirementsの容量が不正です")
    requests.append({"url": url, "retrieved_at": _now(), "size_bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()})
    return data


def _json(url, opener, requests, cancelled):
    try:
        data = json.loads(_read(url, opener, requests, cancelled))
    except (UnicodeError, json.JSONDecodeError):
        raise SetupError("配布元APIのJSONを確認できません") from None
    if not isinstance(data, dict):
        raise SetupError("配布元APIにJSON objectが必要です")
    return data


def _archive(url, directory, opener, requests, cancelled, progress):
    _cancelled(cancelled)
    path = None
    try:
        with _open(url, opener) as response:
            expected = _size(response, MAX_ARCHIVE_BYTES)
            with tempfile.NamedTemporaryFile(mode="wb", prefix="mosaic-source-", suffix=".zip",
                                             dir=directory, delete=False) as output:
                path = Path(output.name)
                digest, count = hashlib.sha256(), 0
                started = time.monotonic()
                while True:
                    _cancelled(cancelled)
                    if time.monotonic() - started > DOWNLOAD_TIMEOUT:
                        raise SetupError("コードarchiveの取得が時間制限を超えました")
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > MAX_ARCHIVE_BYTES:
                        raise SetupError("コードarchiveの取得容量が上限を超えています")
                    output.write(block)
                    digest.update(block)
                    progress(phase="最新コードを取得", downloaded=count, total=expected)
                output.flush()
                os.fsync(output.fileno())
        _cancelled(cancelled)
        if not count or (expected is not None and expected != count) or not zipfile.is_zipfile(path):
            raise SetupError("コードarchiveの容量またはZIP形式が不正です")
        record = {"url": url, "retrieved_at": _now(), "size_bytes": count,
                  "sha256": digest.hexdigest(), "cache_path": str(path)}
        requests.append(record.copy())
        return record
    except BaseException:
        if path is not None:
            # Only this call's exclusively created temporary file is removed.
            path.unlink(missing_ok=True)
        raise


def _pins(text, requirements):
    found = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not re.match(r"comfy[-_](?:kitchen|aimdo)(?:$|[\s<>=!~;\[])", line, re.I):
            continue
        match = PIN.fullmatch(line)
        if not match:
            raise SetupError("最新ComfyUIのkitchen / aimdoに対応する固定pinを確認できません")
        name = match[1].lower().replace("_", "-")
        if name in found:
            raise SetupError("最新ComfyUIのkitchen / aimdo指定が重複しています")
        found[name] = name + "==" + match[2]
    if set(found) != {"comfy-kitchen", "comfy-aimdo"}:
        raise SetupError("最新ComfyUIのkitchen / aimdo固定pinが不足しています")
    result, applied = [], set()
    for requirement in requirements:
        match = re.match(r"(comfy[-_](?:kitchen|aimdo))(?:$|[\s<>=!~;\[])", requirement, re.I)
        name = match[1].lower().replace("_", "-") if match else None
        if name in found:
            if name not in applied:
                result.append(found[name])
                applied.add(name)
        else:
            result.append(requirement)
    result.extend(found[name] for name in sorted(set(found) - applied))
    return result, found


def _ffmpeg(opener, requests, cancelled, base):
    api = f"https://api.github.com/repos/{FFMPEG_REPOSITORY}/releases/latest"
    release = _json(api, opener, requests, cancelled)
    tag = _stable_release(release)
    name = f"ffmpeg-{tag.removeprefix('v')}-essentials_build.zip"
    if not re.fullmatch(r"ffmpeg-[A-Za-z0-9._+-]+-essentials_build\.zip", name):
        raise SetupError("FFmpegの既知の安定asset名を確認できません")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise SetupError("FFmpeg releaseのassetを確認できません")
    matches = [a for a in assets if isinstance(a, dict) and a.get("name") == name]
    if len(matches) != 1:
        raise SetupError("FFmpeg essentials_build.zipを一意に確認できません")
    asset = matches[0]
    digest, size = asset.get("digest"), asset.get("size")
    if (not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest)
            or type(size) is not int or not 0 < size <= MAX_ARCHIVE_BYTES):
        raise SetupError("FFmpegの公式asset digest / 容量が不明です。旧版へ戻さず停止します")
    url = f"https://github.com/{FFMPEG_REPOSITORY}/releases/download/{quote(tag, safe='')}/{quote(name, safe='')}"
    if asset.get("browser_download_url") != url:
        raise SetupError("FFmpeg assetの公式取得URLを確認できません")
    digest = digest.split(":", 1)[1].lower()
    result = deepcopy(base)
    result.update(id="ffmpeg-windows-" + digest, url=url, sha256=digest, size_bytes=size, tag=tag)
    return result, {"repo": FFMPEG_REPOSITORY, "tag": tag, "asset": name, "url": url,
                    "sha256": digest, "size_bytes": size, "release_api_url": api,
                    "digest_source": "GitHub release asset metadata; not a signature"}


def resolve_sources(base_recipe, cache_dir, cancelled=lambda: False, progress=lambda **kw: None, opener=None):
    """Explicitly resolve latest code/tools; never resolve or replace models."""
    recipe = deepcopy(base_recipe)
    if (not isinstance(recipe, dict) or not isinstance(recipe.get("sources"), list) or not recipe["sources"]
            or not isinstance(recipe.get("ffmpeg"), dict) or not isinstance(recipe.get("requirements"), list)
            or any(not isinstance(r, str) for r in recipe["requirements"])):
        raise SetupError("導入構成のsources / ffmpeg / requirementsが不正です")
    for source in recipe["sources"]:
        if (not isinstance(source, dict) or source.get("repo") not in ALLOWED_REPOSITORIES
                or not isinstance(source.get("id"), str) or not re.fullmatch(r"[a-z0-9-]{1,100}", source["id"])):
            raise SetupError("導入構成の取得repo / IDは許可リスト内である必要があります")
        base_id = source.get("resolution_base_id", source["id"])
        if not isinstance(base_id, str) or not re.fullmatch(r"[a-z0-9-]{1,59}", base_id):
            raise SetupError("導入構成の元IDが不正です")
    if recipe["sources"][0]["repo"] != CORE_REPOSITORY or sum(s["repo"] == CORE_REPOSITORY for s in recipe["sources"]) != 1:
        raise SetupError("先頭に一つの公式ComfyUI coreが必要です")
    _cancelled(cancelled)
    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    requests, sources = [], []
    selected = opener or _official_open
    try:
        for source in recipe["sources"]:
            _cancelled(cancelled)
            repo = source["repo"]
            progress(phase="最新の導入版を確認", model=source["id"])
            if repo == CORE_REPOSITORY:
                api = f"https://api.github.com/repos/{repo}/releases/latest"
                reference = _stable_release(_json(api, selected, requests, cancelled))
                choice = {"selection": "latest-stable-release", "tag": reference}
            else:
                api = f"https://api.github.com/repos/{repo}"
                reference = _reference(_json(api, selected, requests, cancelled).get("default_branch"))
                choice = {"selection": "default-branch-tip", "branch": reference}
            commit_api = f"https://api.github.com/repos/{repo}/commits/{quote(reference, safe='')}"
            revision = _json(commit_api, selected, requests, cancelled).get("sha")
            if not isinstance(revision, str) or not SHA.fullmatch(revision):
                raise SetupError("公式repoの40桁commit SHAを確認できません")
            revision = revision.lower()
            url = f"https://codeload.github.com/{repo}/zip/{revision}"
            downloaded = _archive(url, directory, selected, requests, cancelled, progress)
            source.update(revision=revision, url=url, size_bytes=downloaded["size_bytes"],
                          sha256=downloaded["sha256"], cache_path=downloaded["cache_path"],
                          id=source.get("resolution_base_id", source["id"]) + "-" + revision,
                          resolution_base_id=source.get("resolution_base_id", source["id"]),
                          terms_url=f"https://github.com/{repo}/tree/{revision}")
            sources.append({"repo": repo, **choice, "sha": revision, **downloaded,
                            "api_urls": [api, commit_api], "verification": "SHA-256 measured over HTTPS; not a signature"})
        core = recipe["sources"][0]
        requirements_url = f"https://raw.githubusercontent.com/{CORE_REPOSITORY}/{core['revision']}/requirements.txt"
        raw = _read(requirements_url, selected, requests, cancelled)
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeError:
            raise SetupError("最新ComfyUIのrequirementsをUTF-8として確認できません") from None
        recipe["requirements"], pins = _pins(text, recipe["requirements"])
        recipe["ffmpeg"], ffmpeg = _ffmpeg(selected, requests, cancelled, recipe["ffmpeg"])
    except (OSError, ValueError) as error:
        if isinstance(error, SetupError):
            raise
        raise SetupError("最新の導入構成を取得・検証できません。旧固定版へ戻さず停止します") from None
    _cancelled(cancelled)
    recipe["core_patch"] = None
    recipe["resolution"] = {"schema_version": 1, "resolved_at": _now(), "sources": sources, "ffmpeg": ffmpeg,
                            "requirements": {"url": requirements_url, "sha256": hashlib.sha256(raw).hexdigest(), "pins": pins},
                            "requests": requests, "models_changed": False, "signature_verified": False}
    return recipe
