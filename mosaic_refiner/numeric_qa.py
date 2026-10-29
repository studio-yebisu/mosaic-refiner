"""Numeric checks of RGB buffers; no preview, rendering or semantic inspection."""
from .detection import union
from .native_core import compare_frame
from .shape_masks import source_alpha


def frame_statistics(source, result, width, height, entries, feather, engine=None):
    # Only the authoritative shape union needs a mask. Every other pixel is
    # checked as outside by the Rust traversal; holes remain alpha=0.
    if entries:
        from .geometry import Rect
        roi = union([Rect(**entry["roi"]) for entry in entries])
        alpha = source_alpha(entries, roi, feather)
    else:
        roi = alpha = None
    return compare_frame(source, result, width, height, alpha, roi, engine=engine)
