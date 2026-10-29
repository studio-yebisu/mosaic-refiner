"""Encode explicit source PTS; mux original audio without timestamp normalization."""
from fractions import Fraction
from .mp4_audio_tables import read_movie, movie_scale, preserve_audio, audio_chunks, children, handler, UnsupportedAudioTables

from .media import MediaError, command, probe, read_exact, rgb_reader


def encode_timeline_output(chunks, info, source, target, cancelled):
    import av
    import numpy as np
    temporary = target.with_name(target.stem + ".video" + target.suffix)
    tb = Fraction(info.time_base)
    durations = {pts * tb: duration * tb for pts, duration in zip(info.frame_pts, info.frame_durations)}
    # In a B-frame source, a frame's packet duration can differ from the next
    # presentation timestamp interval. MP4 derives sample durations from DTS,
    # so letting the no-B-frame encoder set DTS=PTS silently changes durations.
    # Build a monotonic decode clock from the recorded durations. Its origin is
    # the latest value that keeps DTS <= PTS for every frame; composition offsets
    # then preserve both the original display clock and all sample durations.
    cumulative, ticks = 0, []
    for duration in info.frame_durations:
        ticks.append(cumulative)
        cumulative += duration
    origin = min(pts - tick for pts, tick in zip(info.frame_pts, ticks))
    decode_times = {pts * tb: (origin + tick) * tb for pts, tick in zip(info.frame_pts, ticks)}
    container = None
    try:
        options = {"video_track_timescale": str(tb.denominator)} if target.suffix.lower() == ".mp4" else {}
        native_audio = target.suffix.lower() == ".mp4" and info.audio_streams > 0 and source.suffix.lower() in (".mp4", ".mov", ".m4a", ".m4v")
        if native_audio:
            try:
                entries, original_movie = read_movie(source)
                tracks = [track for kind, track in children(original_movie) if kind == b"trak"]
                if (sum(handler(track) == b"vide" for track in tracks) != 1
                        or any(handler(track) not in (b"vide", b"soun") for track in tracks)):
                    raise UnsupportedAudioTables("追加trackを持つ入力は通常muxで検証します")
                for kind, track in children(original_movie):
                    if kind == b"trak" and handler(track) == b"soun":
                        audio_chunks(track, entries)
                options["movie_timescale"] = str(movie_scale(original_movie))
            except UnsupportedAudioTables:
                # Unsupported layouts keep the existing packet mux path; its
                # strict output verification still decides whether it is valid.
                native_audio = False
        container = av.open(str(temporary), "w", format="mp4" if target.suffix.lower() == ".mp4" else "nut", options=options)
        codec = "libx264rgb" if target.suffix.lower() == ".mp4" else "ffv1"
        stream = container.add_stream(codec, rate=Fraction(info.fps))
        stream.width, stream.height = info.width, info.height
        stream.pix_fmt = "rgb24" if codec == "libx264rgb" else "bgr0"
        stream.time_base = stream.codec_context.time_base = tb
        stream.options = {"crf": "0", "preset": "ultrafast", "bf": "0"} if codec == "libx264rgb" else {"level": "3"}

        def mux(packet):
            instant = packet.pts * packet.time_base
            if instant not in durations:
                raise MediaError("encoderのpacket PTSが原時刻の外です")
            duration = durations[instant] / packet.time_base
            decode_time = decode_times[instant] / packet.time_base
            if duration.denominator != 1 or decode_time.denominator != 1:
                raise MediaError("encoderが原duration/復号時刻を表現できません")
            packet.dts = int(decode_time)
            packet.duration = int(duration)
            container.mux(packet)

        index, size = 0, info.width * info.height * 3
        for chunk in chunks:
            count = probe(chunk).frame_count
            with rgb_reader(chunk) as reader:
                for _ in range(count):
                    if cancelled():
                        raise KeyboardInterrupt
                    pixels = np.frombuffer(read_exact(reader, size), np.uint8).reshape(info.height, info.width, 3)
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    frame.pts, frame.duration, frame.time_base = info.frame_pts[index], info.frame_durations[index], tb
                    for packet in stream.encode(frame):
                        mux(packet)
                    index += 1
        if index != info.frame_count:
            raise MediaError("原時刻とchunk frame数が一致しません")
        for packet in stream.encode():
            mux(packet)
        container.close()
        container = None
        if cancelled():
            raise KeyboardInterrupt
        if native_audio:
            preserve_audio(temporary, source, target, cancelled)
            return
        args = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-copyts", "-i", str(temporary), "-i", str(source),
                "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-map_chapters", "1", "-c", "copy",
                "-fps_mode:v", "passthrough", "-avoid_negative_ts", "disabled"]
        if target.suffix.lower() == ".mp4":
            args += ["-video_track_timescale", str(tb.denominator), "-f", "mp4"]
        else:
            args += ["-f", "nut"]
        command([*args, str(target)])
    finally:
        if container is not None:
            container.close()
        temporary.unlink(missing_ok=True)
