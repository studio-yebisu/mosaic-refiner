"""Loopback-only WebGUI sharing the resumable ApplicationJob core."""
import json
import os
from pathlib import Path
import secrets
import shutil
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlsplit
import uuid

from .application import job_from_spec, new_job, save_session
from .checkpoint import atomic_json, load_json, digest_file, summarize
from .geometry import Rect
from .media import command
from .processing_options import resolve_options, catalog
from .h3_lora_profiles import available_profiles


class Controller:
    def __init__(self, directory, backend="h3"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.recent = self.directory / "session.json"
        self.backend = backend
        self.lock = threading.RLock()
        self.worker = self.job = None
        self.uploading = False
        self.session = load_json(self.recent) if self.recent.exists() else None
        self.state = {"state": "ready" if self.session else "idle", "phase": "前の動画を再開できます" if self.session else "動画を選択してください"}
        restored = self.completed_result()
        if restored:
            self.state = {"state": "complete", "phase": "完成動画を保存できます", "result": restored}

    def completed_result(self):
        """Reuse a verified saved output independently of today's generation code."""
        if not self.session:
            return None
        spec = self.session["spec"]
        job = Path(spec["job_dir"])
        try:
            manifest = load_json(job / "manifest.json")
            source, output = Path(spec["source"]), Path(spec["output"])
            flags = manifest.get("verification") or {}
            if (manifest.get("state") != "complete" or manifest.get("source") != str(source.resolve())
                    or manifest.get("output") != str(output.resolve())
                    or not all(flags.get(k) for k in ("pts_exact", "tail_duration_exact", "audio_packets_exact", "full_decode"))
                    or manifest.get("output_sha256") != digest_file(output)
                    or manifest.get("config", {}).get("source_sha256") != digest_file(source)):
                return None
            result = summarize(job)
            result["output_integrity_checked"] = True
            return result
        except (OSError, ValueError, KeyError):
            return None

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def snapshot(self):
        with self.lock:
            from .learned_detection import model_path
            from .restoration_models import ready as restoration_ready
            options = (resolve_options(self.session['spec'].get('processing_options')) if self.session
                       else resolve_options())
            return {**self.state, "busy": self.busy(), "uploading": self.uploading,
                    "pipeline_revision": "h3-setup-v5",
                    "processing_presets": catalog(),
                    "h3_lora_profiles": available_profiles(),
                    "session": self.session, "server_time": time.time(),"detector_ready":model_path().is_file(),
                    "restoration_ready": restoration_ready(options.lora_profile)}

    def upload(self, stream, length, name):
        if not length or length < 0:
            raise ValueError("空の動画は読み込めません")
        suffix = Path(name).suffix.lower()
        if suffix not in (".mp4", ".mkv", ".mov", ".nut", ".avi", ".webm"):
            raise ValueError("対応動画: MP4 / MKV / MOV / NUT / AVI / WebM")
        with self.lock:
            if self.busy() or self.uploading:
                raise ValueError("現在の処理が終わってから動画を選択してください")
            self.uploading = True
        folder = self.directory / "inputs" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        partial, source = folder / "upload.partial", folder / ("source" + suffix)
        try:
            if shutil.disk_usage(folder).free < length + 256 * 1024 * 1024:
                raise ValueError("動画の保存に必要な空き容量がありません")
            with partial.open("xb") as target:
                remaining = length
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("動画の転送が途中で切れました。もう一度選択してください")
                    target.write(chunk)
                    remaining -= len(chunk)
            partial.rename(source)
            return self.select_source(source, name, folder.name, copied=True)
        finally:
            # Only our incomplete transfer is disposable; keep uploaded originals.
            partial.unlink(missing_ok=True)
            with self.lock:
                self.uploading = False

    def select_source(self, source, name, identifier, copied=False):
        metadata = json.loads(command(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                      "-show_entries", "stream=width,height:format=duration", "-of", "json", str(source)]))
        streams = metadata.get("streams", [])
        if not streams:
            raise ValueError("映像streamを読み取れません")
        video = streams[0]
        spec = new_job(source, self.directory / "jobs")
        spec.update(backend=self.backend, context_seconds="auto", processing_options=resolve_options().to_dict())
        session = {"spec":spec,"name":name,"width":video["width"],"height":video["height"],
                   "duration":metadata.get("format",{}).get("duration"),"preview":None,
                   "preview_enabled":False,"copied_input":copied,"id":identifier}
        with self.lock:
            self.session = session
            atomic_json(self.recent,session)
            self.state = {"state":"ready","phase":"読み込み完了。自動処理を開始できます"}
        return self.snapshot()

    def local(self, path):
        source = Path(path).expanduser()
        if not source.is_absolute() or str(path).replace(chr(92), '/').startswith('//'):
            raise ValueError("このPC内の動画の絶対パスを指定してください")
        source = source.resolve()
        if source.suffix.lower() not in (".mp4", ".mkv", ".mov", ".nut", ".avi", ".webm") or not source.is_file():
            raise ValueError("指定した動画ファイルを読み取れません")
        with self.lock:
            if self.busy() or self.uploading:
                raise ValueError("現在の処理が終わってから動画を選択してください")
            self.uploading = True
        try:
            return self.select_source(source, source.name, uuid.uuid4().hex)
        finally:
            with self.lock:
                self.uploading = False

    def enable_preview(self):
        # Video pixels are decoded for display only after this explicit action.
        with self.lock:
            if not self.session or self.busy() or self.uploading:
                raise ValueError("動画選択後、処理開始前にプレビューを表示できます")
            folder = self.directory / "previews" / self.session["id"]
            folder.mkdir(parents=True,exist_ok=True)
            target = folder / "preview.png"
            if not target.exists():
                target.write_bytes(command(["ffmpeg","-v","error","-nostdin","-noautorotate",
                    "-i",self.session["spec"]["source"],"-map","0:v:0","-frames:v","1",
                    "-vf","scale=640:360:force_original_aspect_ratio=decrease","-f","image2pipe","-vcodec","png","pipe:1"]))
            self.session.update(preview=str(target),preview_enabled=True)
            atomic_json(self.recent,self.session)
        return self.snapshot()

    def prepare_detector(self):
        from .learned_detection import setup
        with self.lock:
            if self.busy() or self.uploading:
                raise ValueError("処理完了後にモデルを導入してください")
            self.uploading = True
            previous = dict(self.state)
            self.state = {"state":"setup","phase":"自動検出モデルを準備しています"}
        try:
            setup(progress=self.progress)
        finally:
            with self.lock:
                self.uploading = False
                self.state = previous
        return self.snapshot()

    def prepare_restoration(self, settings=None):
        from .restoration_models import setup
        with self.lock:
            if self.busy() or self.uploading:
                raise ValueError("処理完了後にモデルを導入してください")
            self.uploading = True
            previous = dict(self.state)
            self.state = {"state":"setup","phase":"補正モデルを準備しています"}
        try:
            settings = settings or {}
            setup(progress=self.progress, lora_profile=settings.get('lora_profile', 'unlocked_v2'),
                  accept_terms=settings.get('accept_terms') is True)
        finally:
            with self.lock:
                self.uploading = False
                self.state = previous
        return self.snapshot()

    def start(self, settings, resume=False):
        with self.lock:
            if self.busy() or self.uploading:
                raise ValueError("すでに処理中です")
            if not self.session:
                raise ValueError("動画を先に選択してください")
            if resume:
                restored = self.completed_result()
                if restored:
                    self.state = {"state": "complete", "phase": "完成動画を保存できます", "result": restored}
                    return self.snapshot()
            spec = dict(self.session["spec"])
            if not resume:
                options = resolve_options(settings.get("processing_options", spec.get("processing_options"))).to_dict()
                raw_context = settings.get("context_seconds", "auto")
                context = "auto" if raw_context == "auto" else int(raw_context)
                if context not in ("auto",5,10,15):
                    raise ValueError("区間長は自動・5・10・15秒から選択してください")
                roi = settings.get("roi")
                if roi is not None:
                    if not isinstance(roi, dict) or set(roi) != {"x", "y", "width", "height"} or any(type(v) is not int for v in roi.values()):
                        raise ValueError("対象領域は整数の矩形で指定してください")
                    Rect(**roi).validate(self.session["width"], self.session["height"])
                # A settings change starts a fresh job, preserving old checkpoints.
                if (spec.get("context_seconds", 5) != context or spec.get("roi") != roi
                        or resolve_options(spec.get("processing_options")).to_dict() != options
                        or (Path(spec["job_dir"]) / "manifest.json").is_file() or Path(spec["output"]).exists()):
                    fresh = new_job(Path(spec["source"]), self.directory / "jobs")
                    fresh["backend"] = self.backend
                    spec = fresh
                spec.update(context_seconds=context, roi=roi, processing_options=options)
                # The worker marks retention after its manifest exists; the core
                # refuses nonempty, unowned job directories.
                spec["keep_intermediates"] = True
            spec["keep_intermediates"] = True
            self.session["spec"] = spec
            atomic_json(self.recent, self.session)
            save_session(spec, self.directory / "jobs" / "recent.json")
            self.state = {"state": "running", "phase": "入力の時刻・音声を確認しています", "started_at": time.time()}
            self.job = job_from_spec(spec, self.progress)
            self.worker = threading.Thread(target=self.execute, daemon=False)
            self.worker.start()
        return self.snapshot()

    def progress(self, **event):
        with self.lock:
            if self.session and self.session["spec"].get("keep_intermediates"):
                job = Path(self.session["spec"]["job_dir"])
                if (job / "manifest.json").is_file() and not (job / "keep-intermediates.json").exists():
                    atomic_json(job / "keep-intermediates.json",{"keep":True})
            if "phase" in event and event["phase"] != self.state.get("phase"):
                for key in ("frame", "total", "elapsed_seconds", "node_stage", "prepass_frame", "prepass_total"):
                    self.state.pop(key, None)
            self.state.update(event)
            self.state["updated_at"] = time.time()

    def execute(self):
        try:
            result = self.job.execute()
            phase = "処理完了。補正結果の見た目をご確認ください" if result["state"] == "complete" else "停止しました。再開できます"
            with self.lock:
                self.state.update(state=result["state"], phase=phase, result=result,
                                  finished_at=time.time(),wall_seconds=round(time.time()-self.state["started_at"],2))
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state.update(state="error", phase=str(exc), finished_at=time.time(),
                                  wall_seconds=round(time.time()-self.state["started_at"],2))

    def cancel(self):
        with self.lock:
            if self.busy():
                self.job.cancel()
                self.state["phase"] = "取消し要求を受け付けました。現在の窓を保存して停止します"
        return self.snapshot()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, controller):
        if address[0] != "127.0.0.1":
            raise ValueError("WebGUIは127.0.0.1だけで公開します")
        self.controller = controller
        self.token = secrets.token_urlsafe(32)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        # Apply the browser boundary to HTML, JSON, files and error responses.
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def send_bytes(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, data, status=200):
        self.send_bytes(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def valid_host(self):
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def valid_origin(self):
        origins = self.headers.get_all("Origin", [])
        # Keep local CLI clients working; a supplied browser origin must match.
        return not origins or (len(origins) == 1 and origins[0] in (
            f"http://127.0.0.1:{self.server.server_port}",
            f"http://localhost:{self.server.server_port}"))

    def valid_token(self, media=False):
        values = self.headers.get_all('X-Mosaic-Token', [])
        if values:
            return len(values) == 1 and secrets.compare_digest(
                values[0].encode('utf-8'), self.server.token.encode('ascii'))
        if media:
            from urllib.parse import parse_qs
            return parse_qs(urlsplit(self.path).query).get('token') == [self.server.token]
        return False

    def do_GET(self):
        # The root page contains no capability. Only the private startup URL
        # supplies it, in a fragment that the browser does not send to servers.
        if not self.valid_host():
            return self.json({"error": "localhostから開いてください"}, 403)
        path = urlsplit(self.path).path
        if path.startswith('/api/') and not self.valid_token(path in ('/api/preview', '/api/output')):
            return self.json({'error': 'start-web.cmdからWebGUIを開き直してください'}, 403)
        c = self.server.controller
        if path == "/":
            page = Path(__file__).with_name("web.html").read_text(encoding="utf-8")
            page = page.replace("__PROCESSING_PRESETS__", json.dumps(catalog(), ensure_ascii=False))
            self.send_bytes(page.encode(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self.json(c.snapshot())
        elif path in ("/api/preview", "/api/output"):
            session = c.snapshot()["session"]
            if not session:
                return self.json({"error": "動画が選択されていません"}, 404)
            if path.endswith("preview") and not session.get("preview_enabled"):
                return self.json({"error":"プレビューは非表示です。画面の表示ボタンから開いてください"},409)
            file = Path(session["preview"] if path.endswith("preview") else session["spec"]["output"])
            if path.endswith("output") and c.snapshot()["state"] != "complete":
                return self.json({"error": "出力はまだ完成していません"}, 409)
            if not file.is_file():
                return self.json({"error": "ファイルが見つかりません"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "image/png" if path.endswith("preview") else "application/octet-stream")
            self.send_header("Content-Length", str(file.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            if path.endswith("output"):
                name = Path(session["name"]).stem + ".refined" + file.suffix
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(name))
            self.end_headers()
            with file.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile, 1024 * 1024)
        else:
            self.json({"error": "見つかりません"}, 404)

    def do_POST(self):
        if not self.valid_host() or not self.valid_token():
            self.close_connection = True
            return self.json({"error": "WebGUIを開き直してください"}, 403)
        if not self.valid_origin():
            self.close_connection = True
            return self.json({"error": "このPCのWebGUIから操作してください"}, 403)
        path = urlsplit(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if path == "/api/upload":
                name = unquote(self.headers.get("X-Filename", "video.mp4")).replace("\\", "/").rsplit("/", 1)[-1]
                result = self.server.controller.upload(self.rfile, length, name)
            else:
                if length < 0 or length > 65536:
                    raise ValueError("設定リクエストが大きすぎます")
                settings = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(settings, dict):
                    raise ValueError("設定はJSON objectが必要です")
                if path == "/api/start":
                    result = self.server.controller.start(settings)
                elif path == "/api/resume":
                    result = self.server.controller.start(settings, resume=True)
                elif path == "/api/cancel":
                    result = self.server.controller.cancel()
                elif path == "/api/local":
                    result = self.server.controller.local(settings.get("path",""))
                elif path == "/api/show-preview":
                    result = self.server.controller.enable_preview()
                elif path == "/api/prepare-detector":
                    result = self.server.controller.prepare_detector()
                elif path == "/api/prepare-restoration":
                    result = self.server.controller.prepare_restoration(settings)
                else:
                    raise ValueError("操作が見つかりません")
            self.json(result)
        except Exception as exc:
            self.close_connection = True
            self.json({"error": str(exc)}, 400)


def main(port=8765, directory=None):
    directory = Path(directory) if directory is not None else Path(
        os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "MosaicRefiner/web"
    server = Server(("127.0.0.1", port), Controller(directory))
    print(f"WebGUI: http://127.0.0.1:{server.server_port}/#token={server.token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.controller.cancel()
    finally:
        if server.controller.worker:
            server.controller.worker.join()
        server.server_close()
