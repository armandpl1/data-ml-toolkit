"""Check that repositories use the tool versions pinned by this toolkit.

The toolkit's `dev` extra pins one version of each shared tool (ruff, mypy,
pytest, pre-commit). A repo that depends on `data-ml-toolkit[dev]` gets those
versions in its uv.lock. This module reads each repo's uv.lock and reports
any tool whose locked version differs, plus which toolkit version it uses.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib.metadata import requires
from pathlib import Path

from packaging.requirements import Requirement

TOOLKIT = "data-ml-toolkit"


def pinned_tools() -> dict[str, str]:
    """{package: version} for every `==` pin in the toolkit's `dev` extra."""
    pins: dict[str, str] = {}
    for line in requires(TOOLKIT) or []:
        req = Requirement(line)
        if req.marker is None or not req.marker.evaluate({"extra": "dev"}):
            continue
        for spec in req.specifier:
            if spec.operator == "==":
                pins[req.name] = spec.version
    return pins


@dataclass(frozen=True)
class RepoReport:
    repo: str
    toolkit: str | None  # version or tag of data-ml-toolkit in uv.lock, None if absent
    locked: dict[str, str | None]  # tool -> locked version (None if not in uv.lock)
    drift: dict[str, tuple[str | None, str]]  # tool -> (locked, pinned) where they differ

    @property
    def in_sync(self) -> bool:
        return self.toolkit is not None and not self.drift


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(lock_path: Path) -> tuple[dict[str, str], str | None]:
    """({package: version}, toolkit version-or-tag) from a uv.lock file."""
    with lock_path.open("rb") as f:
        lock = tomllib.load(f)
    versions: dict[str, str] = {}
    toolkit: str | None = None
    for pkg in lock.get("package", []):
        name = _normalize(pkg["name"])
        versions[name] = pkg.get("version", "")
        if name == TOOLKIT:
            source = pkg.get("source", {})
            tag = re.search(r"[?&]tag=([^&#]+)", source.get("git", ""))
            toolkit = tag.group(1) if tag else pkg.get("version")
    return versions, toolkit


def check_repo(repo: Path, pins: dict[str, str]) -> RepoReport:
    lock = repo / "uv.lock"
    if not lock.exists():
        raise FileNotFoundError(f"{repo} has no uv.lock")
    versions, toolkit = locked_versions(lock)
    locked = {tool: versions.get(tool) for tool in pins}
    drift = {
        tool: (locked[tool], pinned)
        for tool, pinned in pins.items()
        if locked[tool] is not None and locked[tool] != pinned
    }
    return RepoReport(repo=repo.name, toolkit=toolkit, locked=locked, drift=drift)
