"""Build mediamtx path entries for each enabled inference job.

v1 skeleton: returns {} until backend implementations land. The plugin's
__init__.render_paths() calls this; the host's core/renderer.py merges
the dict into mediamtx.yml.
"""
from __future__ import annotations

from typing import Any

from . import jobs as jobs_mod


def build_paths(ctx) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for j in jobs_mod.list_jobs(ctx):
        if not j.enabled:
            continue
        # Backend implementations land in subsequent slices.
        # For now, no path is emitted — the job exists in YAML but produces
        # no mediamtx output.
    return paths
