"""Plan ownership of CFR source frames without loading video or model weights."""

from dataclasses import asdict, dataclass
from fractions import Fraction
from math import ceil
from bisect import bisect_left


@dataclass(frozen=True)
class SaveUnit:
    index: int
    first_frame: int
    end_frame_exclusive: int


def timeline_units(pts, time_base, unit_seconds, scene_cuts=()):
    """Partition indices by actual display timestamps and scene boundaries."""
    times = [Fraction(p) * Fraction(time_base) for p in pts]
    boundaries = {0, len(times), *scene_cuts}
    start = times[0]
    previous = 0
    for index, instant in enumerate(times):
        bucket = (instant - start) // unit_seconds
        if bucket != previous:
            boundaries.add(index)
            previous = bucket
    ordered = sorted(boundaries)
    return tuple(SaveUnit(index, first, end) for index, (first, end) in enumerate(zip(ordered, ordered[1:])) if first < end)


@dataclass(frozen=True)
class ProcessingPlan:
    stage: str
    source_frame_count: int
    source_fps: str
    save_unit_seconds: str
    context_seconds: int
    model_fps: int
    candidate_context_frames: int
    units: tuple[SaveUnit, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def plan_cfr(
    frame_count: int,
    fps: Fraction,
    unit_seconds: Fraction = Fraction(15),
    context_seconds: int = 5,
) -> ProcessingPlan:
    """Assign every source frame exactly once to a completed-output unit.

    These units do not describe inference overlap or model execution. The H3
    context frame count is an untested candidate derived from its public grid.
    Source timestamps stay in the source FPS; they are never replaced by 24fps.
    """
    if frame_count < 0:
        raise ValueError("入力フレーム数は0以上にしてください")
    if fps <= 0 or unit_seconds <= 0:
        raise ValueError("FPSと保存単位の秒数は正数にしてください")
    if context_seconds not in (5, 10, 15):
        raise ValueError("初期context候補は5・10・15秒です")

    units = []
    first = 0
    while first < frame_count:
        # Skip empty sub-frame intervals directly, including extremely small
        # positive unit durations. Runtime is bounded by source frame count.
        boundary_index = Fraction(first) // (unit_seconds * fps) + 1
        end = min(frame_count, ceil(boundary_index * unit_seconds * fps))
        units.append(SaveUnit(len(units), first, end))
        first = end

    raw_frames = context_seconds * 24
    aligned_frames = raw_frames + (5 - raw_frames) % 17
    return ProcessingPlan(
        stage="planning-only",
        source_frame_count=frame_count,
        source_fps=f"{fps.numerator}/{fps.denominator}",
        save_unit_seconds=str(unit_seconds),
        context_seconds=context_seconds,
        model_fps=24,
        candidate_context_frames=aligned_frames,
        units=tuple(units),
    )


@dataclass(frozen=True)
class ModelWindow:
    index: int
    scene_first_frame: int
    scene_end_frame_exclusive: int
    model_first_frame: int
    valid_model_frames: int
    padded_model_frames: int
    model_fps: int = 24

    def source_frame(self, local_index: int, source_fps: Fraction) -> int:
        """Nearest source sample; pad with this scene's last valid sample."""
        if local_index < 0 or local_index >= self.padded_model_frames:
            raise ValueError("window内のframe番号が範囲外です")
        local_index = min(local_index, self.valid_model_frames - 1)
        relative = Fraction(self.model_first_frame + local_index, self.model_fps) * source_fps
        nearest = (relative + Fraction(1, 2)).numerator // (relative + Fraction(1, 2)).denominator
        return min(self.scene_end_frame_exclusive - 1, self.scene_first_frame + nearest)


def model_windows(
    frame_count: int,
    fps: Fraction,
    context_seconds: int = 5,
    overlap_seconds: Fraction = Fraction(1),
    scene_cuts: tuple[int, ...] = (),
) -> tuple[ModelWindow, ...]:
    """CPU-only H3 candidate grid. Never cross a scene or drop the tail."""
    plan_cfr(frame_count, fps, context_seconds=context_seconds)
    if not 0 <= overlap_seconds < context_seconds:
        raise ValueError("overlapは0以上、context未満にしてください")
    if tuple(sorted(set(scene_cuts))) != scene_cuts or any(c <= 0 or c >= frame_count for c in scene_cuts):
        raise ValueError("scene境界は入力範囲内の昇順・重複なしframe番号です")
    valid_limit = context_seconds * 24
    overlap = ceil(overlap_seconds * 24)
    if overlap >= valid_limit:
        raise ValueError("24fpsへ丸めたoverlapがcontext以上です")
    windows = []
    boundaries = (0, *scene_cuts, frame_count)
    for first, end in zip(boundaries, boundaries[1:]):
        model_count = ceil(Fraction(end - first, 1) / fps * 24)
        start = 0
        while start < model_count:
            valid = min(valid_limit, model_count - start)
            padded = valid + (5 - valid) % 17
            windows.append(ModelWindow(len(windows), first, end, start, valid, padded))
            if start + valid >= model_count:
                break
            start += valid_limit - overlap
    return tuple(windows)


def model_windows_timeline(pts, durations, time_base, context_seconds=5,
                           overlap_seconds=Fraction(1), scene_cuts=()):
    """Model context by actual scene duration, without inferring time from FPS."""
    if not 0 <= overlap_seconds < context_seconds or context_seconds not in (5, 10, 15):
        raise ValueError("context/overlapが不正です")
    if tuple(sorted(set(scene_cuts))) != tuple(scene_cuts) or any(c <= 0 or c >= len(pts) for c in scene_cuts):
        raise ValueError("scene境界が不正です")
    windows = []
    boundaries = (0, *scene_cuts, len(pts))
    tb = Fraction(time_base)
    limit, overlap = context_seconds * 24, ceil(overlap_seconds * 24)
    for first, end in zip(boundaries, boundaries[1:]):
        stop = pts[end] * tb if end < len(pts) else (pts[-1] + durations[-1]) * tb
        count = ceil((stop - pts[first] * tb) * 24)
        start = 0
        while start < count:
            valid = min(limit, count - start)
            windows.append(ModelWindow(len(windows), first, end, start, valid, valid + (5 - valid) % 17))
            if start + valid >= count:
                break
            start += limit - overlap
    return tuple(windows)
