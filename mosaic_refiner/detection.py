"""Conservative CPU pixel-block candidates, scene cuts, and bounded tracks.

This detects regular flat color blocks, not semantic objects or lost detail.
"""
from bisect import bisect_right
from dataclasses import asdict
from fractions import Fraction

from .geometry import Rect
from .media import read_exact, rgb_reader


def union(rects):
    x, y = min(r.x for r in rects), min(r.y for r in rects)
    return Rect(x, y, max(r.x + r.width for r in rects) - x,
                max(r.y + r.height for r in rects) - y)


def overlap(a, b):
    area = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x)) * max(
        0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    return area / max(1, a.width * a.height + b.width * b.height - area)


def detect_blocks(rgb, width, height, block_sizes=(8, 12, 16, 24, 32)):
    import numpy as np
    pixels = np.frombuffer(rgb, np.uint8).reshape(height, width, 3).astype(np.float32)
    dx = np.mean(abs(np.diff(pixels, axis=1)), axis=2)
    dy = np.mean(abs(np.diff(pixels, axis=0)), axis=2)
    # Reuse exact integer pixel moments across grid phases instead of
    # re-scanning every pixel for each cell standard deviation.
    moments = np.concatenate((pixels, pixels * pixels), axis=2).astype(np.float64)
    integral = np.zeros((height + 1, width + 1, 6), np.float64)
    integral[1:, 1:] = moments.cumsum(axis=0).cumsum(axis=1)
    candidates = []
    for size in block_sizes:
        if min(width, height) < size * 3:
            continue
        # Grid phase proposals from edge energy; include phase zero.
        x_scores = [dx[:, offset::size].sum() for offset in range(size)]
        y_scores = [dy[offset::size, :].sum() for offset in range(size)]
        xs = {0, *((int(i) + 1) % size for i in np.argsort(x_scores)[-2:])}
        ys = {0, *((int(i) + 1) % size for i in np.argsort(y_scores)[-2:])}
        for top in ys:
            for left in xs:
                rows, cols = (height - top) // size, (width - left) // size
                if min(rows, cols) < 3:
                    continue
                grid = integral[top:top + rows * size + 1:size,
                                left:left + cols * size + 1:size]
                stats = (grid[1:, 1:] - grid[:-1, 1:] - grid[1:, :-1] + grid[:-1, :-1]) / (size * size)
                means = stats[:, :, :3]
                deviations = np.sqrt(np.maximum(0, stats[:, :, 3:] - means * means))
                flat = deviations.mean(axis=2) < 3.5
                horizontal = np.zeros((rows, cols), bool)
                vertical = np.zeros((rows, cols), bool)
                h = abs(np.diff(means, axis=1)).mean(axis=2) > 7
                v = abs(np.diff(means, axis=0)).mean(axis=2) > 7
                horizontal[:, :-1] |= h
                horizontal[:, 1:] |= h
                vertical[:-1] |= v
                vertical[1:] |= v
                good = flat & horizontal & vertical
                seen = set()
                for row, col in zip(*np.nonzero(good)):
                    if (row, col) in seen:
                        continue
                    pending, group = [(row, col)], []
                    while pending:
                        point = pending.pop()
                        if point in seen:
                            continue
                        seen.add(point)
                        r, c = point
                        if not (0 <= r < rows and 0 <= c < cols and good[r, c]):
                            continue
                        group.append(point)
                        pending.extend(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
                    if len(group) < 9:
                        continue
                    r0, r1 = min(p[0] for p in group), max(p[0] for p in group) + 1
                    c0, c1 = min(p[1] for p in group), max(p[1] for p in group) + 1
                    if len(group) / ((r1 - r0) * (c1 - c0)) < .65:
                        continue
                    candidates.append(Rect(int(left + c0 * size), int(top + r0 * size),
                                           int((c1 - c0) * size), int((r1 - r0) * size)))
    # Keep all disjoint candidates; merge duplicates and overlapping blocks.
    merged = []
    for candidate in candidates:
        hits = [r for r in merged if overlap(r, candidate) > .05]
        if hits:
            merged = [r for r in merged if r not in hits]
            candidate = union([candidate, *hits])
        merged.append(candidate)
    return tuple(sorted(merged, key=lambda r: (r.y, r.x)))


class RegionSchedule:
    def __init__(self, records, scene_cuts=(), max_gap_frames=12, metadata=None):
        self.records = records
        self.scene_cuts = tuple(scene_cuts)
        self.max_gap = max_gap_frames
        self.frames = sorted(records)
        self.metadata = metadata or {}

    def at(self, frame):
        return tuple(Rect(**r["roi"]) for r in self.entries_at(frame))

    def entries_at(self, frame):
        index = bisect_right(self.frames, frame) - 1
        if index < 0:
            return ()
        previous = self.frames[index]
        # Conservative hold until next observation; no long gap or scene bridge.
        if frame - previous > self.max_gap or any(previous < c <= frame for c in self.scene_cuts):
            return ()
        return tuple(self.records[previous])

    def to_dict(self):
        return {"method": self.metadata.get("method", "regular-flat-block-heuristic-v1"), "quality_adopted": False,
                "metadata": self.metadata,
                "scene_cuts": list(self.scene_cuts), "max_gap_frames": self.max_gap,
                "records": {str(k): v for k, v in self.records.items()}}


def analyze(source, info, cancelled=lambda: False, progress=lambda **kw: None, detector_identity=None):
    if detector_identity is None:
        from .learned_detection import identity
        detector_identity = identity()
    if detector_identity.get("learned_model_ready"):
        return analyze_tracked(source, info, cancelled, progress, detector_identity)
    import numpy as np
    stride = max(1, round(float(Fraction(info.fps)) / 6))
    records, cuts, tracks = {}, [], {}
    last_thumb = None
    next_id = 0
    with rgb_reader(source) as reader:
        for frame_id in range(info.frame_count):
            if cancelled():
                raise KeyboardInterrupt
            frame = read_exact(reader, info.width * info.height * 3)
            pixels = np.frombuffer(frame, np.uint8).reshape(info.height, info.width, 3)
            thumb = pixels[::max(1, info.height // 36), ::max(1, info.width // 64)].astype(float)
            cut = last_thumb is not None and np.mean(abs(thumb - last_thumb)) / 255 > .32
            last_thumb = thumb
            if cut:
                cuts.append(frame_id)
                tracks = {}
            if frame_id % stride and not cut:
                continue
            boxes = detect_blocks(frame, info.width, info.height)
            available = dict(tracks)
            current = {}
            entries = []
            for roi in boxes:
                matches = [(overlap(roi, old), track) for track, (old, seen) in available.items()
                           if frame_id - seen <= stride * 2]
                score, track = max(matches, default=(0, -1))
                if score < .15:
                    track, next_id = next_id, next_id + 1
                else:
                    available.pop(track)
                current[track] = (roi, frame_id)
                entries.append({"track_id": track, "roi": asdict(roi), "confidence": "heuristic"})
            tracks = current
            records[frame_id] = entries
            progress(phase="対象領域を調査", frame=frame_id + 1, total=info.frame_count)
    return RegionSchedule(records, cuts, stride)


def track_translation(previous, current, roi, width, height):
    """Bounded, forward/backward checked LK translation; never bridge a cut."""
    import cv2
    import numpy as np
    sx, sy = previous.shape[1] / width, previous.shape[0] / height
    mask = np.zeros(previous.shape, np.uint8)
    x0, y0 = max(0, int((roi.x - 16)*sx)), max(0, int((roi.y - 16)*sy))
    x1, y1 = min(mask.shape[1], int((roi.x + roi.width + 16)*sx)), min(mask.shape[0], int((roi.y + roi.height + 16)*sy))
    mask[y0:y1, x0:x1] = 255
    points = cv2.goodFeaturesToTrack(previous, maxCorners=48, qualityLevel=.02, minDistance=3, mask=mask)
    if points is None or len(points) < 4:
        return roi, False
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(21,21), maxLevel=2)
    back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None, winSize=(21,21), maxLevel=2)
    valid = (status[:,0] != 0) & (reverse_status[:,0] != 0) & (np.linalg.norm(back[:,0] - points[:,0], axis=1) < 1.5)
    if valid.sum() < 4:
        return roi, False
    delta = moved[valid,0] - points[valid,0]
    displacement = np.median(delta, axis=0)
    if np.median(np.linalg.norm(delta - displacement, axis=1)) > 3:
        return roi, False
    x, y = round(roi.x + displacement[0]/sx), round(roi.y + displacement[1]/sy)
    if abs(x-roi.x) > width*.15 or abs(y-roi.y) > height*.15:
        return roi, False
    x, y = min(width-roi.width, max(0, x)), min(height-roi.height, max(0, y))
    return Rect(x, y, roi.width, roi.height), True


def analyze_tracked(source, info, cancelled, progress, detector_identity):
    import cv2
    import numpy as np
    import time
    from .learned_detection import Detector
    started = time.monotonic()
    if detector_identity.get("provider") == "DmlExecutionProvider":
        from .h3_runtime import api
        # Share the GPU only after the canonical inference queue is empty.
        while True:
            if cancelled():
                raise KeyboardInterrupt
            queue = api("/queue")
            if not queue.get("queue_running") and not queue.get("queue_pending"):
                break
            progress(phase="GPUの空きを待っています（対象検出）")
            time.sleep(1)
    from .bounded_stream import BoundedStream
    detector = Detector(detector_identity.get("score_threshold", .4),
                        batch_size=detector_identity.get("batch_size", 1))
    stride = max(1, round(float(Fraction(info.fps)) / detector_identity.get("observations_per_second", 3)))
    scale = min(1, 512/max(info.width, info.height))
    shape = (max(1, round(info.width*scale)), max(1, round(info.height*scale)))
    tracks, records, cuts, counts = {}, {}, [], {}
    previous, next_id = None, 0
    detector_seconds = flow_seconds = 0.
    observations = 0
    def decode():
        thumb = None
        with rgb_reader(source) as reader:
            for frame_id in range(info.frame_count):
                if cancelled():
                    raise KeyboardInterrupt
                frame = read_exact(reader, info.width*info.height*3)
                pixels = np.frombuffer(frame,np.uint8).reshape(info.height,info.width,3)
                gray = cv2.cvtColor(cv2.resize(pixels,shape,interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
                small = cv2.resize(gray,(64,36)).astype(np.float32)
                cut = thumb is not None and np.mean(abs(small-thumb))/255 > .32
                thumb = small
                observed = frame_id % stride == 0 or cut
                yield {"frame": frame_id, "gray": gray, "cut": cut,
                       "tensor": detector.preprocess(pixels) if observed else None}

    decoded = BoundedStream(decode, capacity=detector_identity.get("decode_queue_capacity", 8),
                            name="mosaic-decode", cancelled=cancelled)

    def infer():
        nonlocal detector_seconds, observations
        pending, sampled = [], []
        for item in decoded:
            pending.append(item)
            if item["tensor"] is not None:
                sampled.append(item)
            if len(sampled) < detector.batch_size and item["frame"] != info.frame_count-1:
                continue
            if sampled:
                t = time.monotonic()
                predictions = detector.predict_batch([s["tensor"] for s in sampled],
                                                     [(info.height, info.width)] * len(sampled))
                detector_seconds += time.monotonic()-t
                observations += len(sampled)
                for sample, boxes in zip(sampled, predictions):
                    sample["boxes"] = boxes
                    sample["tensor"] = None
            yield pending
            pending, sampled = [], []

    inferred = BoundedStream(infer, capacity=detector_identity.get("inference_queue_capacity", 1),
                             name="mosaic-detector", cancelled=cancelled)
    from contextlib import ExitStack
    from itertools import chain
    with ExitStack() as stack:
        stack.callback(detector.close)
        stack.enter_context(decoded)
        stack.enter_context(inferred)
        for item in chain.from_iterable(inferred):
            if cancelled():
                raise KeyboardInterrupt
            frame_id, gray, cut = item["frame"], item["gray"], item["cut"]
            if cut:
                cuts.append(frame_id)
                tracks = {}
            t = time.monotonic()
            if previous is not None and not cut:
                for entry in tracks.values():
                    entry["roi"], entry["flow_valid"] = track_translation(previous,gray,entry["roi"],info.width,info.height)
            flow_seconds += time.monotonic()-t
            if "boxes" in item:
                available = dict(tracks)
                current = {}
                for roi, score, shape_mask in item["boxes"]:
                    matches = [(overlap(roi, old["roi"]), track) for track, old in available.items()]
                    value, track = max(matches, default=(0, -1))
                    if value < .15:
                        track, next_id = next_id, next_id+1
                    else:
                        available.pop(track)
                    counts[track] = counts.get(track,0)+1
                    current[track] = {"roi":roi,"seen":frame_id,"score":score,"flow_valid":True,
                                      "shape_mask": shape_mask}
                # Brief prediction only, bounded by two detector observations.
                for track, entry in available.items():
                    if frame_id-entry["seen"] < stride*2 and entry.get("flow_valid"):
                        current[track] = entry
                tracks = current
            records[frame_id] = [{"track_id":track,"roi":asdict(e["roi"]),"confidence":e["score"],
                                  "predicted":frame_id != e["seen"], "shape_mask": e["shape_mask"]}
                                 for track,e in tracks.items() if frame_id-e["seen"] < stride*2]
            previous = gray
            if frame_id % stride == 0 or frame_id == info.frame_count-1:
                progress(phase="対象領域を検出・追跡",frame=frame_id+1,total=info.frame_count,
                         detector=detector_identity["method"], detector_batch=detector.batch_size)
    # Reject one-observation weak transients before any diffusion is submitted.
    records = {f:[e for e in entries if counts[e["track_id"]] >= 2 or e["confidence"] >= .85]
               for f,entries in records.items()}
    metadata = {**detector_identity,"analysis_seconds":time.monotonic()-started,
                "detector_seconds":detector_seconds,"tracking_seconds":flow_seconds,"observations":observations,
                "queues": {"decode": decoded.stats, "inference": inferred.stats}}
    return RegionSchedule(records,cuts,0,metadata)
