"""Fixed-size, scene-local crop translation with bounded smoothing."""
from dataclasses import asdict

from .detection import union
from .geometry import Rect


def follow_crop_path(schedule, width, height, first, end, margin=32):
    boxes = {frame: union(regions) for frame in range(first, end)
             if (regions := schedule.at(frame))}
    if not boxes:
        return {}
    cw = min(width, max(r.width for r in boxes.values()) + 2 * margin)
    ch = min(height, max(r.height for r in boxes.values()) + 2 * margin)
    path, previous, last = {}, None, None
    for frame in range(first, end):
        roi = boxes.get(frame)
        if roi is None:
            if previous is not None:
                path[frame] = asdict(previous)
            last = None  # do not smooth across an unobserved gap
            continue
        x, y = roi.x + (roi.width - cw) / 2, roi.y + (roi.height - ch) / 2
        if last is not None and previous is not None:
            x, y = previous.x + .25 * (x - previous.x), previous.y + .25 * (y - previous.y)
        # Keep the entire target and its available context inside the crop,
        # even after an abrupt translation, multiple targets, or at a border.
        xlo = min(width - cw, max(0, roi.x + roi.width + margin - cw))
        xhi = max(0, min(width - cw, roi.x - margin))
        ylo = min(height - ch, max(0, roi.y + roi.height + margin - ch))
        yhi = max(0, min(height - ch, roi.y - margin))
        previous = Rect(round(min(xhi, max(xlo, x))), round(min(yhi, max(ylo, y))), cw, ch)
        previous.validate(width, height)
        path[frame], last = asdict(previous), frame
    # Context frames before the first observation use its crop; their mask is 0.
    initial = path[min(path)]
    for frame in range(first, min(path)):
        path[frame] = initial
    return path
