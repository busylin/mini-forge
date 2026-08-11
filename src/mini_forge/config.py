from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    root = path.parent.parent
    config["project_root"] = str(root)
    for key, value in config.get("paths", {}).items():
        candidate = Path(value)
        if not candidate.is_absolute():
            config["paths"][key] = str((root / candidate).resolve())
    for override in overrides or []:
        dotted_key, raw_value = override.split("=", 1)
        cursor = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    return config

