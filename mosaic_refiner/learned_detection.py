"""Independent RF-DETR ONNX adapter; no third-party application source is used."""
import os
from pathlib import Path

from .checkpoint import digest_file
from .geometry import Rect
from .model_setup import Model, install


class DetectorModel(Model):
    @property
    def url(self):
        return "https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.onnx"


MODEL = DetectorModel(
    id="rfdetr-v6-large", repo="Kruk2/jasna", revision="0.1", filename="rfdetr-v6-large.onnx",
    target="rfdetr-v6-large.onnx", size_bytes=149272820,
    sha256="e8c1af4b1d7be2b99ef21325a8140b7bea15132df0b25cd30d32128bcd8cd644",
    license_label="Apache-2.0 (Jasna作者の専用detector条件)",
    terms_url="https://github.com/Kruk2/jasna/blob/79d9f11582c03accc16197896780241c54ba2823/LICENSING.md",
    source_checked_at="2026-10-26",
)


def model_directory():
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "MosaicRefiner/detector"


def model_path():
    installed = MODEL.destination(model_directory())
    development = Path(__file__).resolve().parents[1] / "models" / MODEL.filename
    return installed if installed.exists() else development


def setup(cancelled=lambda: False, progress=lambda **kw: None):
    # Called only by an explicit setup operation; never at import/start/inference.
    from .model_setup import default_cache
    return install(model_directory(), default_cache(), models=[MODEL], cancelled=cancelled, progress=progress)


def identity():
    p = model_path()
    if not p.is_file():
        return {"method": "regular-flat-block-heuristic-v1", "learned_model_ready": False}
    if p.stat().st_size != MODEL.size_bytes or digest_file(p) != MODEL.sha256:
        raise ValueError("自動検出モデルのhashが固定版と一致しません。別ファイルに置換せず導入を確認してください")
    import onnxruntime as ort
    provider = "DmlExecutionProvider" if "DmlExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
    return {"method": "rfdetr-v6-large-onnx", "provider":provider,"runtime_version":ort.__version__,"learned_model_ready": True,
            "model_sha256": MODEL.sha256, "score_threshold": .4, "observations_per_second": 3,
            "tracking": "opencv-lk-forward-backward", "shape_masks": True,
            "batch_size": 2, "decode_queue_capacity": 8, "inference_queue_capacity": 1, "version": 2}


class Detector:
    def __init__(self, threshold=.4, batch_size=1):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if not 1 <= batch_size <= 8:
            raise ValueError("検出batchは1〜8で指定してください")
        options.add_free_dimension_override_by_name("batch", batch_size)
        provider = "DmlExecutionProvider" if "DmlExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
        self.session = ort.InferenceSession(str(model_path()), sess_options=options, providers=[provider,"CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [provider])
        self.options = ort.RunOptions()
        self.options.only_execute_path_to_fetches = True
        self.threshold = threshold
        self.batch_size = batch_size
        shape = self.session.get_inputs()[0].shape
        if shape[1:] != [3, 768, 768]:
            raise ValueError("RF-DETRの入力寸法が固定版と一致しません")

    @staticmethod
    def preprocess(pixels):
        import cv2
        import numpy as np
        small = cv2.resize(pixels, (768, 768), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        small = (small - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return np.ascontiguousarray(small.transpose(2, 0, 1))

    def predict_batch(self, tensors, sizes, with_masks=True):
        import numpy as np
        if not tensors or len(tensors) > self.batch_size or len(tensors) != len(sizes):
            raise ValueError("検出batchの入力が設定と一致しません")
        count = len(tensors)
        padded = tensors + [tensors[-1]] * (self.batch_size - count)
        names = ["dets", "labels", "masks"] if with_masks else ["dets", "labels"]
        values = self.session.run(names, {"input": np.stack(padded)}, self.options)
        return [self._decode(values[0][i], values[1][i], sizes[i],
                             values[2][i] if with_masks else None) for i in range(count)]

    def _decode(self, boxes, logits, size, masks=None):
        import cv2
        import numpy as np
        from .shape_masks import pack_mask
        h, w = size
        scores = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        scores = scores.max(axis=-1)
        selected = []
        from .detection import overlap
        for index in np.argsort(-scores):
            if scores[index] < self.threshold:
                break
            cx, cy, rw, rh = boxes[index]
            x0, y0 = max(0, int((cx-rw/2)*w)), max(0, int((cy-rh/2)*h))
            x1, y1 = min(w, int(np.ceil((cx+rw/2)*w))), min(h, int(np.ceil((cy+rh/2)*h)))
            if x1 <= x0 or y1 <= y0:
                continue
            r = Rect(x0, y0, x1-x0, y1-y0)
            if not any(overlap(r, item[0]) > .5 for item in selected):
                packed = None
                if masks is not None:
                    # Resize logits before applying sigmoid(0)=0.5 threshold.
                    plane = cv2.resize(masks[index], (w, h), interpolation=cv2.INTER_LINEAR)
                    binary = (plane[y0:y1, x0:x1] > 0).astype(np.uint8)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    if not binary.any():
                        continue  # A box without a shape is not a restoration target.
                    packed = pack_mask(binary)
                selected.append((r, float(scores[index]), packed))
        return selected

    def __call__(self, pixels):
        values = self.predict_batch([self.preprocess(pixels)], [pixels.shape[:2]], with_masks=False)[0]
        return [(box, score) for box, score, _ in values]

    def close(self):
        # Drop the DirectML session before any BasicVSR++/H3 window starts.
        self.session = None
