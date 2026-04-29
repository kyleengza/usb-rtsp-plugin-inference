"""CPU backend for the inference worker.

Runs YOLOv8 ONNX via cv2.dnn — no torch / ultralytics dependency. The
.onnx files for this backend are fetched into
/var/cache/usb-rtsp/inference-models/ by the pi-bringup-inference.sh
script (slice 6).

Output format expected from a vanilla YOLOv8 ONNX export:
  shape (1, 84, 8400)  → 4 box (cx,cy,w,h) + 80 class scores per anchor
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2  # type: ignore
import numpy as np  # type: ignore


@dataclass
class Detection:
    class_id: int
    label: str
    score: float
    box: tuple[float, float, float, float]


class CpuBackend:
    """YOLOv8 ONNX via OpenCV DNN."""

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
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        # Stay on CPU — Pi 5 has no usable GPU/Vulkan path for cv2.dnn.
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def close(self) -> None:
        self.net = None  # type: ignore[assignment]

    def __enter__(self) -> "CpuBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        iw, ih = self.input_size
        # YOLOv8 expects RGB float32 NCHW, scaled to 0-1.
        blob = cv2.dnn.blobFromImage(
            frame_bgr, scalefactor=1.0 / 255.0,
            size=(iw, ih), swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()
        # outputs shape: (1, 84, 8400) — transpose to (8400, 84) for easy slicing.
        if outputs.ndim == 3 and outputs.shape[0] == 1:
            preds = outputs[0].T  # (8400, 84)
        else:
            preds = outputs.reshape(-1, outputs.shape[-2]).T
        # Split: first 4 cols = box (cx, cy, w, h) in input-image pixels;
        # next 80 = class scores (sigmoid-activated already in YOLOv8 head).
        boxes_raw = preds[:, :4]
        scores_raw = preds[:, 4:]
        class_ids = scores_raw.argmax(axis=1)
        confidences = scores_raw[np.arange(scores_raw.shape[0]), class_ids]
        # Threshold mask
        mask = confidences >= self.threshold
        if not mask.any():
            return []
        boxes_raw = boxes_raw[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask].astype(np.float32)
        # Convert (cx, cy, w, h) in input-pixel coords → (x, y, w, h)
        # and scale to source-frame coords for cv2.dnn.NMSBoxes.
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
