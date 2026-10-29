"""Source-pixel ROI and a diagnostic compositor; no learned restoration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    def validate(self, frame_width: int, frame_height: int) -> None:
        if (self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0
                or self.x + self.width > frame_width or self.y + self.height > frame_height):
            raise ValueError("ROIは映像内の x,y,width,height（正の幅・高さ）で指定してください")


def crop_rgb(frame: bytes, width: int, height: int, roi: Rect) -> bytes:
    roi.validate(width, height)
    if len(frame) != width * height * 3:
        raise ValueError("RGB frameのbyte数が一致しません")
    return b"".join(frame[((roi.y + y) * width + roi.x) * 3:
                          ((roi.y + y) * width + roi.x + roi.width) * 3] for y in range(roi.height))


def composite_rgb(frame: bytes, width: int, height: int, roi: Rect, restored: bytes, feather: int = 0) -> bytes:
    roi.validate(width, height)
    if len(frame) != width * height * 3 or len(restored) != roi.width * roi.height * 3:
        raise ValueError("原frameまたは補正ROIの寸法が一致しません")
    if feather < 0 or feather > min(roi.width, roi.height) // 2:
        raise ValueError("featherは0以上、ROI短辺の半分以下です")
    from .native_core import composite_rectangle_if_available
    native = composite_rectangle_if_available(frame, width, height, roi, restored, feather)
    if native is not None:
        return native
    result = bytearray(frame)
    for y in range(roi.height):
        for x in range(roi.width):
            distance = min(x, y, roi.width - 1 - x, roi.height - 1 - y)
            weight = 1.0 if not feather else min(1.0, distance / feather)
            offset = ((roi.y + y) * width + roi.x + x) * 3
            crop_offset = (y * roi.width + x) * 3
            for channel in range(3):
                original = frame[offset + channel]
                result[offset + channel] = round(original + weight * (restored[crop_offset + channel] - original))
    return bytes(result)


class DiagnosticCompositor:
    """Explicit identity/invert test backend. Pixels outside ROI are untouched.

    A real backend will provide restored ROI pixels instead of inverting them.
    The feather is inside the rectangle; it never enlarges the write mask.
    """

    def __init__(self, width: int, height: int, roi: Rect, backend: str, feather: int = 0):
        roi.validate(width, height)
        if backend not in ("identity", "invert"):
            raise ValueError("CPU backendはidentityまたはinvertです（AI補正ではありません）")
        if feather < 0 or feather > min(roi.width, roi.height) // 2:
            raise ValueError("featherは0以上、ROI短辺の半分以下です")
        self.width, self.height, self.roi = width, height, roi
        self.backend, self.feather = backend, feather

    def apply(self, frame: bytes, active: bool = True) -> bytes:
        if len(frame) != self.width * self.height * 3:
            raise ValueError("RGB frameのbyte数が一致しません")
        if self.backend == "identity" or not active:
            return frame
        region = crop_rgb(frame, self.width, self.height, self.roi)
        restored = bytes(255 - value for value in region)
        return composite_rgb(frame, self.width, self.height, self.roi, restored, self.feather)
