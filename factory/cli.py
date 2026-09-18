"""Command dispatch.

Each script is also runnable standalone:
    python -m factory.train --config configs/arm_d.yaml

This module hosts the shared `--config` handling, including resolution of the
`extends:` key that merges an arm config over configs/base.yaml.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged: dict[str, Any] = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _resolve_reference(value: str, config: Mapping[str, Any]) -> Any:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        ref = config
        for part in key.split("."):
            if not isinstance(ref, dict) or part not in ref:
                raise KeyError(f"Unknown config reference: {key}")
            ref = ref[part]
        return str(ref)

    return _VAR_PATTERN.sub(replace, value)


def _interpolate_values(obj: Any, config: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {key: _interpolate_values(value, config) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_values(item, config) for item in obj]
    if isinstance(obj, str):
        return _resolve_reference(obj, config)
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config file, resolving local ``extends`` entries and variable refs."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path} must parse to a dictionary")

    parent = raw.get("extends")
    merged: dict[str, Any] = {}
    if parent:
        parent_path = (config_path.parent / str(parent)).resolve()
        merged = load_config(parent_path)
        merged = _deep_merge(merged, raw)
    else:
        merged = dict(raw)

    merged.pop("extends", None)
    merged = _interpolate_values(merged, merged)
    return merged


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Factory CLI")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Arm: {config.get('arm', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
