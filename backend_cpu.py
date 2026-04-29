"""CPU backend for the inference worker.

Uses onnxruntime (apt: python3-onnxruntime) to run YOLOv8/YOLO11 ONNX
exports — cv2.dnn 4.10 can't parse the Concat layer shape these
models use, hence the dedicated runtime.

Output decoding handles the standard YOLO detect-head shape
(1, 84, 8400) — 4 box (cx,cy,w,h) + 80 class scores per anchor.
Same input/output contract as backend_hailo.HailoBackend so the
worker swaps backends with one --backend flag.
"""
from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore  # used only for resize/NMS, not inference
import numpy as np  # type: ignore
import onnxruntime as ort  # type: ignore

from dets import Detection  # type: ignore


class CpuBackend:
    """YOLOv8/YOLO11 ONNX via onnxruntime."""

    def __init__(
        self,
        onnx_path: Path,
        labels: list[str],
        threshold: float = 0.4,
        nms_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
        allow_class_ids: set[int] | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = input_size  # (w, h)
        self.labels = labels
        self.allow_class_ids = allow_class_ids
        # Single-threaded session by default — leave headroom for ffmpeg
        # encode + frame I/O. Pi 5 has 4 cores; onnxruntime defaulting to
        # 4 inter-op threads steals from encoder.
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]

    def __enter__(self) -> "CpuBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        iw, ih = self.input_size
        # Letterbox would be more accurate but plain resize is close enough
        # for a fallback path; backends don't share boxes anyway.
        resized = cv2.resize(frame_bgr, (iw, ih))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        nchw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        nchw = np.expand_dims(nchw, axis=0)

        outputs = self.session.run([self.output_name], {self.input_name: nchw})
        preds = outputs[0]
        # preds shape: (1, 84, 8400) for COCO detect head — drop batch + transpose.
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0].T  # (8400, 84)
        else:
            preds = preds.reshape(-1, preds.shape[-2]).T

        boxes_raw = preds[:, :4]
        scores_raw = preds[:, 4:]
        class_ids = scores_raw.argmax(axis=1)
        confidences = scores_raw[np.arange(scores_raw.shape[0]), class_ids]
        mask = confidences >= self.threshold
        if not mask.any():
            return []
        boxes_raw = boxes_raw[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask].astype(np.float32)

        # Convert (cx,cy,w,h) input-pixel → (x,y,w,h) source-pixel.
        sx = w / float(iw)
        sy = h / float(ih)
        cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        x = (cx - bw / 2) * sx
        y = (cy - bh / 2) * sy
        bw = bw * sx
        bh = bh * sy
        rects = np.stack([x, y, bw, bh], axis=1).tolist()
        keep_idx = cv2.dnn.NMSBoxes(rects, confidences.tolist(),
                                    self.threshold, self.nms_threshold)
        if keep_idx is None or len(keep_idx) == 0:
            return []
        keep_idx = np.array(keep_idx).flatten()

        detections: list[Detection] = []
        for i in keep_idx:
            cid = int(class_ids[i])
            if self.allow_class_ids is not None and cid not in self.allow_class_ids:
                continue
            xi, yi, wi, hi = rects[i]
            detections.append(Detection(
                class_id=cid,
                label=(self.labels[cid] if cid < len(self.labels) else f"class_{cid}"),
                score=float(confidences[i]),
                box=(float(xi), float(yi), float(xi + wi), float(yi + hi)),
            ))
        return detections
