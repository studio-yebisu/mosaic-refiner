"""Copy ISO BMFF audio samples and edit/sample tables without demux/remux.

Some MOV edit lists expose duplicate final AAC packets. A normal packet remux
forces monotonic DTS and changes that presentation. Keep the original audio
track tables, relocate only its chunk offsets, and copy only its audio bytes.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import struct


class UnsupportedAudioTables(ValueError):
    pass


@dataclass(frozen=True)
class Box:
    start: int
    size: int
    kind: bytes
    header: int


def file_boxes(stream):
    stream.seek(0, 2)
    end = stream.tell()
    position = 0
    while position < end:
        stream.seek(position)
        header = stream.read(8)
        if len(header) != 8:
            raise UnsupportedAudioTables("containerのbox headerが途中です")
        size, kind = struct.unpack(">I4s", header)
        length = 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) != 8:
                raise UnsupportedAudioTables("拡張box headerが途中です")
            size, length = struct.unpack(">Q", extended)[0], 16
        elif size == 0:
            size = end - position
        if size < length or position + size > end:
            raise UnsupportedAudioTables("containerのbox範囲が不正です")
        yield Box(position, size, kind, length)
        position += size


def body(atom):
    if len(atom) < 8:
        raise UnsupportedAudioTables("boxが短すぎます")
    return atom[16 if atom[:4] == b"\0\0\0\1" else 8:]


def box(kind, payload):
    size = len(payload) + 8
    if size > 0xFFFFFFFF:
        return struct.pack(">I4sQ", 1, kind, size + 8) + payload
    return struct.pack(">I4s", size, kind) + payload


def children(atom):
    import io
    data = body(atom)
    for entry in file_boxes(io.BytesIO(data)):
        yield entry.kind, data[entry.start:entry.start + entry.size]


def one(atom, *path):
    for kind in path:
        found = [data for name, data in children(atom) if name == kind]
        if len(found) != 1:
            raise UnsupportedAudioTables(f"必要なboxが一意ではありません: {kind!r}")
        atom = found[0]
    return atom


def handler(track):
    return body(one(track, b"mdia", b"hdlr"))[8:12]


def track_id(track):
    data = body(one(track, b"tkhd"))
    offset = 20 if data[0] == 1 else 12
    return struct.unpack_from(">I", data, offset)[0]


def movie_scale(moov):
    data = body(one(moov, b"mvhd"))
    offset = 20 if data[0] == 1 else 12
    scale = struct.unpack_from(">I", data, offset)[0]
    if not scale:
        raise UnsupportedAudioTables("movie timescaleが0です")
    return scale


def read_movie(path):
    with Path(path).open("rb") as stream:
        entries = list(file_boxes(stream))
        movies = [entry for entry in entries if entry.kind == b"moov"]
        if len(movies) != 1 or any(entry.kind == b"moof" for entry in entries):
            raise UnsupportedAudioTables("非fragmentedの一意なmoovが必要です")
        entry = movies[0]
        if entry.size > 64 * 1024 * 1024:
            raise UnsupportedAudioTables("moov metadataが上限を超えています")
        stream.seek(entry.start)
        moov = stream.read(entry.size)
    if any(kind in (b"mvex", b"cmov") for kind, _ in children(moov)):
        raise UnsupportedAudioTables("fragmented/compressed moovは未対応です")
    return entries, moov


def audio_chunks(track, entries):
    # External data references must never cause reads outside the source file.
    dref = body(one(track, b"mdia", b"minf", b"dinf", b"dref"))
    references = list(children(box(b"dref", dref[8:])))
    if (not references or len(references) != int.from_bytes(dref[4:8], "big")
            or any(kind not in (b"url ", b"alis") or body(data) != b"\0\0\0\1"
                   for kind, data in references)):
        raise UnsupportedAudioTables("外部音声data referenceは未対応です")
    stbl = one(track, b"mdia", b"minf", b"stbl")
    offsets = [(kind, data) for kind, data in children(stbl) if kind in (b"stco", b"co64")]
    if len(offsets) != 1:
        raise UnsupportedAudioTables("音声chunk offset tableが一意ではありません")
    kind, data = offsets[0]
    payload = body(data)
    count = struct.unpack_from(">I", payload, 4)[0]
    width, fmt = (4, ">I") if kind == b"stco" else (8, ">Q")
    expected = 8 + count * width
    if len(payload) < expected or any(payload[expected:]):
        raise UnsupportedAudioTables("音声offset tableの長さが不正です")
    starts = [struct.unpack_from(fmt, payload, 8 + index * width)[0] for index in range(count)]
    sizes = body(one(stbl, b"stsz"))
    fixed, sample_count = struct.unpack_from(">II", sizes, 4)
    expected = 12 if fixed else 12 + sample_count * 4
    if len(sizes) < expected or any(sizes[expected:]):
        raise UnsupportedAudioTables("音声sample size tableの長さが不正です")
    variable = [] if fixed else [item[0] for item in struct.iter_unpack(">I", sizes[12:expected])]
    mapping = body(one(stbl, b"stsc"))
    row_count = struct.unpack_from(">I", mapping, 4)[0]
    expected = 8 + row_count * 12
    if len(mapping) < expected or any(mapping[expected:]):
        raise UnsupportedAudioTables("音声sample/chunk tableの長さが不正です")
    rows = list(struct.iter_unpack(">III", mapping[8:expected]))
    if not rows or rows[0][0] != 1 or any(not samples or not description for _, samples, description in rows):
        raise UnsupportedAudioTables("音声sample/chunk割当が不正です")
    if any(a[0] >= b[0] for a, b in zip(rows, rows[1:])) or rows[-1][0] > count:
        raise UnsupportedAudioTables("音声chunk割当が昇順ではありません")
    ranges = [(entry.start + entry.header, entry.start + entry.size) for entry in entries if entry.kind == b"mdat"]
    result, row, sample = [], 0, 0
    for number, start in enumerate(starts, 1):
        while row + 1 < len(rows) and rows[row + 1][0] <= number:
            row += 1
        samples = rows[row][1]
        if sample + samples > sample_count:
            raise UnsupportedAudioTables("音声sample数がchunk割当を超えています")
        length = fixed * samples if fixed else sum(variable[sample:sample + samples])
        if not length or not any(low <= start and start + length <= high for low, high in ranges):
            raise UnsupportedAudioTables("音声chunkがmdat範囲の外です")
        result.append((start, length))
        sample += samples
    if sample != sample_count:
        raise UnsupportedAudioTables("音声sampleがchunkへ全数割当されていません")
    return result


def relocate(track, starts):
    containers = (b"trak", b"mdia", b"minf", b"stbl")
    def rewrite(kind, atom):
        if kind in (b"stco", b"co64"):
            # Always use 64-bit offsets, including videos exceeding 4GiB.
            return box(b"co64", body(atom)[:4] + struct.pack(">I", len(starts)) +
                       b"".join(struct.pack(">Q", start) for start in starts))
        if kind in containers:
            return box(kind, b"".join(rewrite(name, data) for name, data in children(atom)))
        return atom
    return rewrite(b"trak", track)


def audio_table_identity(path):
    """Timing/codec/edit tables, excluding physical chunk locations only."""
    _, moov = read_movie(path)
    header = body(one(moov, b"mvhd"))
    duration_offset, fmt = (24, ">Q") if header[0] == 1 else (16, ">I")
    duration = struct.unpack_from(fmt, header, duration_offset)[0]
    property_start = duration_offset + (8 if header[0] == 1 else 4)
    containers = (b"trak", b"mdia", b"minf", b"stbl")
    def canonical(kind, atom):
        if kind in (b"stco", b"co64"):
            return b""
        if kind in containers:
            return box(kind, b"".join(canonical(name, value) for name, value in children(atom)))
        return atom
    tracks = [hashlib.sha256(canonical(kind, atom)).hexdigest()
              for kind, atom in children(moov) if kind == b"trak" and handler(atom) == b"soun"]
    return {"movie_scale": movie_scale(moov), "movie_duration": duration,
            "movie_media_properties": hashlib.sha256(header[property_start:-4]).hexdigest(),
            "audio_track_tables": tracks}


def verify_nullable_aac_durations(source, output, before, after, pcm_hash):
    """Verify absent ffprobe durations using exact tables and decoded PCM.

    This permits no differing payload, PTS, DTS, side data, known duration, or
    stream count. The native edit/sample tables must match byte for byte after
    removing relocated chunk addresses, and every decoded PCM stream matches.
    """
    if not before or len(before) != len(after):
        raise UnsupportedAudioTables("音声stream数が一致しません")
    missing = 0
    for old, new in zip(before, after):
        if old["codec"] != "aac" or {k: v for k, v in old.items() if k != "packets"} != {
                k: v for k, v in new.items() if k != "packets"}:
            raise UnsupportedAudioTables("AAC以外または音声codec情報が一致しません")
        if len(old["packets"]) != len(new["packets"]):
            raise UnsupportedAudioTables("音声packet数が一致しません")
        for a, b in zip(old["packets"], new["packets"]):
            if a == b:
                continue
            if set(a) != set(b) or {key for key in a if a[key] != b[key]} != {"duration"}:
                raise UnsupportedAudioTables("音声packetの内容または時刻が一致しません")
            if a["duration"] is not None and b["duration"] is not None:
                raise UnsupportedAudioTables("既知の音声durationが一致しません")
            missing += 1
    if not missing:
        raise UnsupportedAudioTables("欠落durationの検証対象ではありません")
    identity = audio_table_identity(source)
    if identity != audio_table_identity(output):
        raise UnsupportedAudioTables("元の音声sample/edit時刻表が一致しません")
    hashes = []
    for index in range(len(before)):
        a, b = pcm_hash(source, index), pcm_hash(output, index)
        if a != b:
            raise UnsupportedAudioTables("復号した音声PCMが一致しません")
        hashes.append(a)
    return {"method": "aac-packets-native-tables-pcm-v1", "nullable_packet_durations": missing,
            "audio_timing_tables_exact": True, "decoded_pcm_exact": True,
            "decoded_pcm_sha256": hashes, "native_audio_tables": identity}


def copy_region(source, target, start, length, cancelled):
    source.seek(start)
    while length:
        if cancelled():
            raise KeyboardInterrupt
        data = source.read(min(length, 1024 * 1024))
        if not data:
            raise UnsupportedAudioTables("copy中に入力が途中で終了しました")
        target.write(data)
        length -= len(data)


def preserve_audio(video, source, target, cancelled=lambda: False):
    video, source, target = map(Path, (video, source, target))
    if target.resolve() in (video.resolve(), source.resolve()):
        raise ValueError("原本・生成映像を上書きできません")
    source_entries, source_moov = read_movie(source)
    video_entries, video_moov = read_movie(video)
    if movie_scale(source_moov) != movie_scale(video_moov):
        raise UnsupportedAudioTables("movie timescaleを原本と一致させてください")
    tracks = [data for kind, data in children(source_moov) if kind == b"trak"]
    originals = [track for track in tracks if handler(track) == b"vide"]
    audio = [track for track in tracks if handler(track) == b"soun"]
    if len(originals) != 1 or not audio:
        raise UnsupportedAudioTables("原本は映像1本＋音声が必要です")
    ids = [track_id(track) for track in tracks]
    if len(ids) != len(set(ids)) or not all(ids):
        raise UnsupportedAudioTables("原本のtrack IDが一意ではありません")
    layouts = [audio_chunks(track, source_entries) for track in audio]
    cursor = video.stat().st_size + 16
    relocated = []
    for track, layout in zip(audio, layouts):
        starts = []
        for _, length in layout:
            starts.append(cursor)
            cursor += length
        relocated.append(relocate(track, starts))
    updated = []
    for kind, atom in children(video_moov):
        if kind == b"trak":
            if handler(atom) != b"vide":
                raise UnsupportedAudioTables("生成入力は映像だけにしてください")
            pieces = []
            for name, data in children(atom):
                if name == b"tkhd":
                    payload = bytearray(body(data))
                    struct.pack_into(">I", payload, 20 if payload[0] == 1 else 12, track_id(originals[0]))
                    data = box(name, payload)
                pieces.append(data)
            atom = box(kind, b"".join(pieces))
        elif kind == b"mvhd":
            # MOV's movie clock and selection fields also belong to the source
            # presentation. The new video's track retains its explicit PTS.
            payload = bytearray(body(one(source_moov, b"mvhd")))
            struct.pack_into(">I", payload, len(payload) - 4, max(ids) + 1)
            atom = box(kind, payload)
        elif kind in (b"udta", b"meta"):
            continue
        updated.append(atom)
    metadata = [atom for kind, atom in children(source_moov) if kind in (b"udta", b"meta")]
    final_moov = box(b"moov", b"".join([*updated, *relocated, *metadata]))
    with video.open("rb") as image_bytes, source.open("rb") as audio_bytes, target.open("xb") as output:
        for entry in video_entries:
            if entry.kind == b"moov":
                # Keep the video chunk offsets unchanged even for front moov.
                output.write(struct.pack(">I4sQ", 1, b"free", entry.size))
                left = entry.size - 16
                while left:
                    if cancelled():
                        raise KeyboardInterrupt
                    length = min(left, 1024 * 1024)
                    output.write(bytes(length))
                    left -= length
            else:
                copy_region(image_bytes, output, entry.start, entry.size, cancelled)
        output.write(struct.pack(">I4sQ", 1, b"mdat", cursor - video.stat().st_size))
        for layout in layouts:
            for start, length in layout:
                copy_region(audio_bytes, output, start, length, cancelled)
        output.write(final_moov)
    return {"method": "iso-audio-sample-tables-v1", "audio_tracks": len(audio),
            "audio_chunks": sum(len(layout) for layout in layouts),
            "audio_bytes": sum(length for layout in layouts for _, length in layout)}
