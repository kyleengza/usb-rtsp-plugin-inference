"""Curated model registry — loads etc/models.yml, filters to entries
whose backend files actually exist on disk."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PLUGIN_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = PLUGIN_DIR / "etc" / "models.yml"


@dataclass
class HailoModel:
    name: str
    hef: Path
    post_so: Path
    labels: list[str]
    fps_target: int


@dataclass
class CpuModel:
    name: str
    onnx: Path
    labels: list[str]
    fps_target: int


def _load_raw() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"hailo": {}, "cpu": {}, "labels": {}}
    return yaml.safe_load(REGISTRY_PATH.read_text()) or {}


def _resolve_labels(raw: dict[str, Any], key: str) -> list[str]:
    labels = (raw.get("labels") or {}).get(key) or []
    return [str(x) for x in labels]


def hailo_models() -> list[HailoModel]:
    raw = _load_raw()
    out: list[HailoModel] = []
    for name, spec in (raw.get("hailo") or {}).items():
        hef = Path(spec.get("hef", ""))
        if not hef.exists():
            continue
        # post_so is informational — not required by the Python worker, but
        # kept in the registry for any future gst-tappas backend.
        post = Path(spec.get("post_so", ""))
        out.append(HailoModel(
            name=name, hef=hef, post_so=post,
            labels=_resolve_labels(raw, spec.get("labels", "coco")),
            fps_target=int(spec.get("fps_target", 30)),
        ))
    return out


def cpu_models() -> list[CpuModel]:
    raw = _load_raw()
    out: list[CpuModel] = []
    for name, spec in (raw.get("cpu") or {}).items():
        onnx = Path(spec.get("onnx", ""))
        if not onnx.exists():
            continue
        out.append(CpuModel(
            name=name, onnx=onnx,
            labels=_resolve_labels(raw, spec.get("labels", "coco")),
            fps_target=int(spec.get("fps_target", 5)),
        ))
    return out


def all_labels(label_set: str = "coco") -> list[str]:
    return _resolve_labels(_load_raw(), label_set)


def has_backend(backend: str) -> bool:
    if backend == "hailo":
        return bool(hailo_models())
    if backend == "cpu":
        return bool(cpu_models())
    return False


def find_model(backend: str, name: str) -> HailoModel | CpuModel | None:
    pool = hailo_models() if backend == "hailo" else cpu_models()
    for m in pool:
        if m.name == name:
            return m
    return None
