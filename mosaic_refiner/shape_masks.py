"""Numerical shape masks; no rendering, previews, or frame-image artifacts."""
import base64
from functools import lru_cache
import zlib


def pack_mask(mask):
    import numpy as np
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("shape mask must be a plane")
    return {"width": mask.shape[1], "height": mask.shape[0],
            "bits_zlib": base64.b64encode(zlib.compress(np.packbits(mask > 0).tobytes(), 3)).decode("ascii")}


@lru_cache(maxsize=128)
def _unpack(data, width, height):
    import numpy as np
    if not (0 < width <= 16384 and 0 < height <= 16384):
        raise ValueError("shape mask dimensions are invalid")
    needed = (width * height + 7) // 8
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(data, validate=True), needed + 1)
    if len(raw) != needed or not decoder.eof or decoder.unused_data:
        raise ValueError("shape mask payload has invalid length")
    result = np.unpackbits(np.frombuffer(raw, np.uint8))[:width * height].reshape(height, width)
    result.flags.writeable = False
    return result


def unpack_mask(packed):
    return _unpack(packed["bits_zlib"], packed["width"], packed["height"])


def entry_alpha(entry, feather=8):
    import cv2
    import numpy as np
    roi = entry["roi"]
    width, height = roi["width"], roi["height"]
    edge = min(feather, width // 2, height // 2)
    if entry.get("shape_mask"):
        binary = unpack_mask(entry["shape_mask"])
        if binary.shape != (height, width):
            binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        if not edge:
            return binary.astype(np.float32)
        # Keep feather entirely INSIDE the detected shape, including its holes.
        distance = cv2.distanceTransform(np.pad(binary, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        return np.where(binary, np.clip(distance / edge, 0, 1), 0).astype(np.float32)
    y, x = np.indices((height, width))
    distance = np.minimum(np.minimum(x, width - 1 - x), np.minimum(y, height - 1 - y))
    return np.minimum(1, distance / edge).astype(np.float32) if edge else np.ones((height, width), np.float32)


def source_alpha(entries, roi, feather=8):
    import numpy as np
    result = np.zeros((roi.height, roi.width), np.float32)
    for entry in entries:
        box = entry["roi"]
        x0, y0 = max(roi.x, box["x"]), max(roi.y, box["y"])
        x1 = min(roi.x + roi.width, box["x"] + box["width"])
        y1 = min(roi.y + roi.height, box["y"] + box["height"])
        if x0 >= x1 or y0 >= y1:
            continue
        alpha = entry_alpha(entry, feather)
        view = result[y0 - roi.y:y1 - roi.y, x0 - roi.x:x1 - roi.x]
        np.maximum(view, alpha[y0 - box["y"]:y1 - box["y"], x0 - box["x"]:x1 - box["x"]], out=view)
    return result
