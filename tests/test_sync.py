from pathlib import Path

import pytest

from data_ml_toolkit import __version__
from data_ml_toolkit.cli import main
from data_ml_toolkit.sync import check_repo, locked_versions, pinned_tools

PINS = {"ruff": "0.17.5", "pytest": "9.2.0"}


def write_lock(repo: Path, packages: dict[str, str], toolkit_tag: str | None) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    blocks = [f'[[package]]\nname = "{n}"\nversion = "{v}"\n' for n, v in packages.items()]
    if toolkit_tag:
        blocks.append(
            '[[package]]\nname = "data-ml-toolkit"\nversion = "0.1.0"\n'
            'source = { git = "https://github.com/armandpl1/data-ml-toolkit'
            f'?tag={toolkit_tag}#abc123" }}\n'
        )
    (repo / "uv.lock").write_text("version = 1\n\n" + "\n".join(blocks), encoding="utf-8")
    return repo


def test_pinned_tools_come_from_the_dev_extra() -> None:
    pins = pinned_tools()
    assert {"ruff", "mypy", "pytest", "pre-commit"} <= pins.keys()
    assert "boto3" not in pins  # the aws extra is not a dev pin
    assert all(v[0].isdigit() for v in pins.values())


def test_locked_versions_reads_tag_and_normalizes_names(tmp_path: Path) -> None:
    repo = write_lock(tmp_path / "r", {"Pre_Commit": "4.7.0"}, toolkit_tag="v0.1.0")
    versions, toolkit = locked_versions(repo / "uv.lock")
    assert versions["pre-commit"] == "4.7.0"
    assert toolkit == "v0.1.0"


def test_repo_in_sync(tmp_path: Path) -> None:
    repo = write_lock(tmp_path / "r", {"ruff": "0.17.5", "pytest": "9.2.0"}, "v0.1.0")
    report = check_repo(repo, PINS)
    assert report.in_sync
    assert report.drift == {}


def test_drift_and_missing_toolkit_are_reported(tmp_path: Path) -> None:
    repo = write_lock(tmp_path / "r", {"ruff": "0.17.4"}, toolkit_tag=None)
    report = check_repo(repo, PINS)
    assert not report.in_sync
    assert report.toolkit is None
    assert report.drift == {"ruff": ("0.17.4", "0.17.5")}
    assert report.locked["pytest"] is None  # not used by that repo: not drift


def test_missing_lock_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"uv\.lock"):
        check_repo(tmp_path, PINS)


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pins = pinned_tools()
    good = write_lock(tmp_path / "good", pins, toolkit_tag="v0.1.0")
    bad = write_lock(tmp_path / "bad", {**pins, "ruff": "0.0.1"}, toolkit_tag=None)

    assert main(["check-sync", str(good)]) == 0
    assert "All repos in sync." in capsys.readouterr().out

    assert main(["check-sync", str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert "bad: does not depend on data-ml-toolkit" in out
    assert f"bad: ruff 0.0.1 (toolkit pins {pins['ruff']})" in out

    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__
