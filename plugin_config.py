"""Plugin-level config (paths, defaults). Distinct from jobs.yml which
is per-job. Lives at ~/.config/usb-rtsp/inference/plugin.yml.

Currently holds:
  clips_root: /path/to/clip/storage   # default ~/.cache/usb-rtsp/inference-clips
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CLIPS_ROOT = Path.home() / ".cache" / "usb-rtsp" / "inference-clips"


def _ctx_config_dir(ctx) -> Path:
    plugin = getattr(ctx, "plugin", None)
    if plugin is not None and hasattr(plugin, "config_dir"):
        return Path(plugin.config_dir)
    return Path(ctx.config_dir)


def _config_path(ctx) -> Path:
    return _ctx_config_dir(ctx) / "plugin.yml"


def load(ctx) -> dict:
    p = _config_path(ctx)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text()) or {}
    return raw if isinstance(raw, dict) else {}


def save(ctx, cfg: dict) -> None:
    p = _config_path(ctx)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    tmp.replace(p)


def clips_root(ctx) -> Path:
    raw = load(ctx).get("clips_root")
    if not raw:
        return DEFAULT_CLIPS_ROOT
    return Path(str(raw)).expanduser()


def set_clips_root(ctx, root: str) -> Path:
    """Validate + persist a new clip-storage root. Returns the resolved
    Path. Raises ValueError if the path can't be created or isn't
    writable (so the API can surface a clear 400)."""
    raw = (root or "").strip()
    if not raw:
        raise ValueError("clips_root must not be empty")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        raise ValueError(
            f"clips_root must be an absolute path (got {raw!r}); "
            "use ~ for home or /mnt/... for a mount")
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ValueError(f"cannot create {p}: {e}") from e
    if not _is_writable(p):
        raise ValueError(f"{p} is not writable by the service user")
    cfg = load(ctx)
    cfg["clips_root"] = str(p)
    save(ctx, cfg)
    return p


def _is_writable(p: Path) -> bool:
    """Probe via tempfile create+delete; covers ACL / mount-readonly cases
    that os.access can lie about."""
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(dir=str(p), delete=True):
            return True
    except OSError:
        return False
