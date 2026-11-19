"""YAML config loading and repo-relative paths.

Each repo keeps its own config files and override rules; this module provides
the shared, strict pieces: read a YAML mapping, find the repo root, and
resolve paths written relative to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file that must contain a mapping.

    Raises FileNotFoundError if the file is missing and ValueError if it is
    empty or not a mapping, so a broken config fails loudly at startup.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} is empty or not a mapping")
    return data


def find_repo_root(start: str | Path, marker: str = "pyproject.toml") -> Path:
    """Walk up from `start` to the first folder containing `marker`."""
    here = Path(start).resolve()
    for folder in (here, *here.parents):
        if (folder / marker).exists():
            return folder
    raise FileNotFoundError(f"No {marker} found in {here} or any parent folder")


def resolve_path(root: str | Path, relative: str | Path) -> Path:
    """Resolve a config path against the repo root. Absolute paths are kept."""
    path = Path(relative)
    return path if path.is_absolute() else Path(root) / path
