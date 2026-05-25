# SPDX-License-Identifier: Apache-2.0
"""Model-architecture helpers for cache behavior decisions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_model_config_path(model_id_or_dir: str) -> Path | None:
    path = Path(model_id_or_dir).expanduser()

    # Direct config path.
    if path.name == "config.json" and path.is_file():
        return path

    # Model directory containing config.json.
    if path.is_dir():
        direct_config = path / "config.json"
        if direct_config.is_file():
            return direct_config

        # Handle a parent directory containing exactly one model subdir.
        model_subdirs = [
            subdir
            for subdir in path.iterdir()
            if subdir.is_dir() and (subdir / "config.json").is_file()
        ]
        if len(model_subdirs) == 1:
            return model_subdirs[0] / "config.json"

    return None


def _model_uses_chunked_kv_cache(model_id_or_dir: str) -> bool:
    """Return True when a model is known to use ChunkedKVCache.

    Defaults to True (safe) if model config cannot be resolved/read.
    """
    config_path = _resolve_model_config_path(model_id_or_dir)
    if config_path is None:
        return True

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read model config at %s: %s", config_path, e)
        return True

    candidates: list[str] = []
    model_type = config.get("model_type")
    if isinstance(model_type, str):
        candidates.append(model_type.lower())

    architectures = config.get("architectures")
    if isinstance(architectures, list):
        candidates.extend(
            arch.lower() for arch in architectures if isinstance(arch, str)
        )

    # ChunkedKVCache is currently only used by Llama-4-derived variants.
    return any("llama4" in value or value.startswith("mllama") for value in candidates)
