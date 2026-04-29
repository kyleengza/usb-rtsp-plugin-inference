"""Hailo backend for the inference worker.

Wraps hailo_platform Python bindings into a single object the worker
loop can call once per frame. Assumes the HEF has NMS bundled (true for
all curated yolov8/yolov6 .hef files in the registry).
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2  # type: ignore
import numpy as np  # type: ignore

import hailo_platform as hpt  # type: ignore


@dataclass
class Detection:
    class_id: int
    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in source-frame pixel coords


class HailoBackend:
    """Loads HEF, holds VDevice + activation context, runs inference per
    frame. Outputs are decoded from the bundled NMS into a flat list."""

    def __init__(
        self,
        hef_path: Path,
        labels: list[str],
        threshold: float = 0.4,
        allow_class_ids: set[int] | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.labels = labels
        self.allow_class_ids = allow_class_ids
        self._stack = contextlib.ExitStack()

        self.hef = hpt.HEF(str(hef_path))
        self.device = self._stack.enter_context(hpt.VDevice())
        cfg_params = hpt.ConfigureParams.create_from_hef(
            self.hef, interface=hpt.HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(self.hef, cfg_params)[0]

        in_info = self.hef.get_input_vstream_infos()[0]
        out_info = self.hef.get_output_vstream_infos()[0]
        self.input_name = in_info.name
        self.output_name = out_info.name
        # Input shape from HEF is (H, W, C) for NHWC.
        ih, iw, ic = in_info.shape
        self.input_size = (int(iw), int(ih))  # cv2.resize wants (w, h)
        self.input_channels = int(ic)

        in_params = hpt.InputVStreamParams.make_from_network_group(
            self.network_group, format_type=hpt.FormatType.UINT8)
        out_params = hpt.OutputVStreamParams.make_from_network_group(
            self.network_group, format_type=hpt.FormatType.FLOAT32)

        self.infer = self._stack.enter_context(
            hpt.InferVStreams(self.network_group, in_params, out_params))
        self._activated = self._stack.enter_context(
            self.network_group.activate(self.network_group.create_params()))

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> "HailoBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        nhwc = np.expand_dims(rgb, axis=0)  # (1, H, W, 3)

        outputs = self.infer.infer({self.input_name: nhwc})
        # Hailo NMS-by-class output (tf_nms_format=False) is wrapped in a
        # batch dim:  outer = list[batch_size], inner = list[num_classes],
        # each inner element ndarray of shape (num_dets, 5) where columns
        # = [y_min, x_min, y_max, x_max, score], coords normalised to [0,1].
        raw = outputs[self.output_name]
        if isinstance(raw, list) and len(raw) >= 1:
            per_class = raw[0]
        elif isinstance(raw, np.ndarray) and raw.ndim >= 1 and raw.shape[0] == 1:
            per_class = raw[0]
        else:
            per_class = raw
        detections: list[Detection] = []
        for class_id, class_dets in enumerate(per_class):
            if class_dets is None or len(class_dets) == 0:
                continue
            if self.allow_class_ids is not None and class_id not in self.allow_class_ids:
                continue
            for det in class_dets:
                if len(det) < 5:
                    continue
                y_min, x_min, y_max, x_max, score = (float(det[i]) for i in range(5))
                if score < self.threshold:
                    # Per-class output is sorted by score desc; can break.
                    break
                detections.append(Detection(
                    class_id=class_id,
                    label=(self.labels[class_id] if class_id < len(self.labels)
                           else f"class_{class_id}"),
                    score=score,
                    box=(x_min * w, y_min * h, x_max * w, y_max * h),
                ))
        return detections
