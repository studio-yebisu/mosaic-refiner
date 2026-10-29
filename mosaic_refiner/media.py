"""FFmpeg adapter for an intentionally narrow, exact-CFR CPU prototype."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

CHILD_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)

class MediaError(ValueError):
    pass


def command(args: list[str]) -> bytes:
    try:
        result = subprocess.run(args, capture_output=True, check=False, creationflags=CHILD_FLAGS)
    except FileNotFoundError as exc:
        raise MediaError(f"実行ファイルがありません: {args[0]}") from exc
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace")[-3000:]
        raise MediaError(f"{Path(args[0]).name}に失敗しました: {message}")
    return result.stdout


def tool_version(name: str) -> str:
    return command([name, "-version"]).decode("utf-8", errors="replace").splitlines()[0]


def require_tools() -> None:
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise MediaError(f"{name}が見つかりません。start-setup.cmdの初回セットアップで導入してください")


def _probe(path: Path, *options: str) -> dict:
    return json.loads(command(["ffprobe", "-v", "error", *options, "-of", "json", str(path)]))


def frame_records(path: Path) -> list[dict]:
    return _probe(path, "-select_streams", "v:0", "-show_frames", "-show_entries",
                  "frame=pts,best_effort_timestamp,duration,pkt_duration,interlaced_frame,width,height,pix_fmt")["frames"]


@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    frame_count: int
    fps: str
    time_base: str
    audio_streams: int
    pixel_format: str
    color_space: str
    color_transfer: str
    frame_pts: tuple[int, ...] = ()
    frame_durations: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["frame_pts"] = list(self.frame_pts)
        data["frame_durations"] = list(self.frame_durations)
        return data


def probe(path: Path, allow_timeline: bool = False) -> MediaInfo:
    """Require exact zero-origin CFR, 8-bit SDR, progressive square pixels.

    Quantized CFR timestamps that do not fit the declared grid are also rejected;
    the prototype must not quietly replace an input's real PTS with a new grid.
    """
    require_tools()
    if not path.is_file():
        raise MediaError("入力動画が見つかりません")
    data = _probe(path, "-show_streams")
    streams = data.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    if len(videos) != 1 or any(s.get("codec_type") not in ("video", "audio") for s in streams):
        raise MediaError("CPU版は映像1本＋音声のみ対応です。字幕・添付・複数映像は未対応です")
    video = videos[0]
    if video.get("disposition", {}).get("attached_pic"):
        raise MediaError("静止画添付streamは未対応です")
    rotation = video.get("tags", {}).get("rotate", 0)
    rotations = [rotation, *(s.get("rotation", 0) for s in video.get("side_data_list", []))]
    if any(float(value) % 360 for value in rotations):
        raise MediaError("回転metadata付き映像は未対応です")
    if video.get("sample_aspect_ratio", "1:1") not in ("1:1", "N/A", "0:1"):
        raise MediaError("正方pixel以外のSARは未対応です")
    allowed = {"rgb24", "bgr24", "bgr0", "rgb0", "gbrp", "gray", "yuv420p", "yuv422p", "yuv444p",
               "yuvj420p", "yuvj422p", "yuvj444p"}
    pixel_format = video.get("pix_fmt", "")
    if pixel_format not in allowed:
        raise MediaError("CPU版はalphaなし8bit映像のみ対応です")
    transfer = video.get("color_transfer", "unknown")
    space = video.get("color_space", "unknown")
    if transfer in ("smpte2084", "arib-std-b67") or "bt2020" in space:
        raise MediaError("HDR/BT.2020は未対応です")
    frames = frame_records(path)
    if not frames:
        raise MediaError("映像frameがありません")
    try:
        time_base = Fraction(video["time_base"])
        try:
            fps = Fraction(video["avg_frame_rate"])
        except (ValueError, ZeroDivisionError):
            try:
                fps = Fraction(video["r_frame_rate"])
            except (KeyError, ValueError, ZeroDivisionError):
                # FFmpeg reports 0/0 for some short lossless checkpoints.
                duration = frames[0].get("duration", frames[0].get("pkt_duration"))
                if len(frames) != 1 or not duration:
                    raise
                fps = 1 / (duration * time_base)
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise MediaError("有効なFPS/time_baseがありません") from exc
    if fps <= 0 or time_base <= 0:
        raise MediaError("FPS/time_baseは正数が必要です")
    width, height = video["width"], video["height"]
    for index, frame in enumerate(frames):
        pts = frame.get("pts")
        if pts is None or (not allow_timeline and Fraction(pts) * time_base != Fraction(index, 1) / fps):
            raise MediaError(f"frame {index}: 原PTSが0始まりの厳密CFRではありません。VFR・offset・時刻丸めは未対応です")
        duration = frame.get("duration", frame.get("pkt_duration"))
        if duration is None or duration <= 0 or (not allow_timeline and Fraction(duration) * time_base != 1 / fps):
            raise MediaError(f"frame {index}: frame durationが不明または一定ではありません")
        if frame.get("interlaced_frame"):
            raise MediaError("interlace映像は未対応です")
        if frame.get("width") != width or frame.get("height") != height or frame.get("pix_fmt") != pixel_format:
            raise MediaError("途中で寸法/pixel形式が変わる映像は未対応です")
        if allow_timeline and index and pts <= frames[index - 1]["pts"]:
            raise MediaError("表示PTSが昇順でない動画は処理できません")
    return MediaInfo(width, height, len(frames), str(fps), str(time_base),
                     sum(s.get("codec_type") == "audio" for s in streams), pixel_format, space, transfer,
                     tuple(f["pts"] for f in frames) if allow_timeline else (),
                     tuple(f.get("duration", f.get("pkt_duration")) for f in frames) if allow_timeline else ())


def read_exact(stream, size: int) -> bytes:
    blocks = bytearray()
    while len(blocks) < size:
        block = stream.read(size - len(blocks))
        if not block:
            break
        blocks.extend(block)
    if len(blocks) != size:
        raise MediaError(f"復号frameが途中で終了しました（{len(blocks)}/{size} bytes）")
    return bytes(blocks)


@contextmanager
def rgb_reader(path: Path, first_frame: int = 0, info: MediaInfo | None = None):
    # Select after decode: accurate, bounded memory, no approximate keyframe seek.
    args = ["ffmpeg", "-v", "error", "-nostdin", "-noautorotate"]
    if first_frame and info is not None:
        instant = (info.frame_pts[first_frame] * Fraction(info.time_base) if info.frame_pts
                   else Fraction(first_frame) / Fraction(info.fps))
        microseconds = instant * 1000000
        ticks = microseconds.numerator // microseconds.denominator
        absolute = abs(ticks)
        text = ("-" if ticks < 0 else "") + f"{absolute // 1000000}.{absolute % 1000000:06d}"
        args += ["-copyts", "-seek_timestamp", "1", "-noaccurate_seek", "-ss", text]
    args += ["-i", str(path), "-map", "0:v:0"]
    if first_frame and info is not None:
        target = info.frame_pts[first_frame] if info.frame_pts else int(instant / Fraction(info.time_base))
        args += ["-vf", f"select=gte(pts\\,{target})"]
    elif first_frame:
        args += ["-vf", f"select=gte(n\\,{first_frame})"]
    args += ["-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=errors, creationflags=CHILD_FLAGS)
        try:
            yield process.stdout
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()


class ChunkEncoder:
    def __init__(self, path: Path, info: MediaInfo):
        self.errors = tempfile.TemporaryFile()
        # Timeline checkpoints store an ordinal sequence, not source timestamps.
        # A fractional average rate can be rounded by the NUT muxer and fail our
        # strict checkpoint check. Use an exact internal grid; final mux restores
        # every source PTS/duration from MediaInfo, independently of this grid.
        checkpoint_rate = "24" if info.frame_pts else info.fps
        args = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-video_size", f"{info.width}x{info.height}", "-framerate", checkpoint_rate, "-i", "pipe:0",
                "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-fps_mode", "passthrough",
                "-f", "nut", str(path)]
        try:
            self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=self.errors, creationflags=CHILD_FLAGS)
        except BaseException:
            self.errors.close()
            raise

    def write(self, frame: bytes) -> None:
        try:
            self.process.stdin.write(frame)
        except BrokenPipeError as exc:
            raise MediaError("chunk encoderが終了しました") from exc

    def close(self, abort: bool = False) -> None:
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            if abort and self.process.poll() is None:
                self.process.terminate()
            code = self.process.wait()
            if code and not abort:
                self.errors.seek(0)
                raise MediaError(self.errors.read().decode("utf-8", errors="replace")[-3000:])
        finally:
            self.errors.close()


def encode_output(chunks: list[Path], info: MediaInfo, source: Path, target: Path, cancelled) -> None:
    """Stream lossless chunks to one encoder: CFR grid never depends on concat rounding."""
    if info.frame_pts:
        from .timeline_media import encode_timeline_output
        return encode_timeline_output(chunks, info, source, target, cancelled)
    args = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-copyts", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-video_size", f"{info.width}x{info.height}", "-framerate", info.fps, "-i", "pipe:0", "-i", str(source),
            "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-map_chapters", "1", "-c:a", "copy",
            "-fps_mode:v", "passthrough", "-avoid_negative_ts", "disabled"]
    if target.suffix.lower() == ".mp4":
        args += ["-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast", "-pix_fmt", "rgb24",
                 "-video_track_timescale", str(Fraction(info.fps).numerator), "-f", "mp4"]
    else:
        args += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-f", "nut"]
    args += [str(target)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=errors, creationflags=CHILD_FLAGS)
        failed = True
        try:
            size = info.width * info.height * 3
            for chunk in chunks:
                if cancelled():
                    raise KeyboardInterrupt
                count = probe(chunk).frame_count
                with rgb_reader(chunk) as reader:
                    for _ in range(count):
                        if cancelled():
                            raise KeyboardInterrupt
                        process.stdin.write(read_exact(reader, size))
            process.stdin.close()
            code = process.wait()
            if code:
                errors.seek(0)
                raise MediaError(errors.read().decode("utf-8", errors="replace")[-3000:])
            failed = False
        except BrokenPipeError as exc:
            errors.seek(0)
            raise MediaError("最終muxに失敗しました: " + errors.read().decode("utf-8", errors="replace")[-3000:]) from exc
        finally:
            if failed:
                if process.poll() is None:
                    process.terminate()
                process.wait()
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass


def audio_signature(path: Path) -> list[dict]:
    data = _probe(path, "-select_streams", "a", "-show_streams", "-show_packets", "-show_data_hash", "sha256",
                  "-show_entries", "stream=index,time_base,codec_name,sample_rate,channels:packet=stream_index,pts,dts,duration,data_hash:packet_side_data")
    result = []
    for stream in data.get("streams", []):
        tb = Fraction(stream["time_base"])
        packets = []
        for packet in data.get("packets", []):
            if packet["stream_index"] == stream["index"]:
                packets.append({
                    "sha256": packet.get("data_hash"),
                    "pts": str(Fraction(packet["pts"]) * tb) if "pts" in packet else None,
                    "dts": str(Fraction(packet["dts"]) * tb) if "dts" in packet else None,
                    "duration": str(Fraction(packet["duration"]) * tb) if "duration" in packet else None,
                    "side_data": packet.get("side_data_list", []),
                })
        result.append({"codec": stream["codec_name"], "sample_rate": stream.get("sample_rate"),
                       "channels": stream.get("channels"), "packets": packets})
    return result


def verify_output(source: Path, output: Path, info: MediaInfo) -> dict:
    final = probe(output, allow_timeline=bool(info.frame_pts))
    if (final.width, final.height, final.frame_count) != (info.width, info.height, info.frame_count):
        raise MediaError("出力の寸法/frame数/FPSが一致しません")
    if info.frame_pts:
        source_tb, target_tb = Fraction(info.time_base), Fraction(final.time_base)
        if ([p * source_tb for p in info.frame_pts] != [p * target_tb for p in final.frame_pts]
                or [p * source_tb for p in info.frame_durations] != [p * target_tb for p in final.frame_durations]):
            raise MediaError("出力が原PTS/durationを正確に保持できません。このcontainerへの出力は未確定です")
    elif Fraction(final.fps) != Fraction(info.fps):
        raise MediaError("出力FPSが原FPSと一致しません")
    before, after = audio_signature(source), audio_signature(output)
    audio_proof = {"method": "packet-signatures-v1"}
    if before != after:
        from .mp4_audio_tables import verify_nullable_aac_durations, UnsupportedAudioTables
        def pcm_hash(path, index):
            return command(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(path),
                            "-map", f"0:a:{index}", "-c:a", "pcm_s32le", "-f", "hash",
                            "-hash", "sha256", "-"]).decode().strip()
        try:
            audio_proof = verify_nullable_aac_durations(source, output, before, after, pcm_hash)
        except UnsupportedAudioTables as exc:
            raise MediaError("音声packetの内容/時刻/durationが一致しません。出力は未確定です") from exc
    command(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(output), "-map", "0", "-f", "null", "-"])
    return {"frame_count": info.frame_count, "fps": info.fps, "pts_exact": True,
            "tail_duration_exact": True, "audio_packets_exact": True, "full_decode": True,
            "ai_restoration": False, "audio_verification": audio_proof}
