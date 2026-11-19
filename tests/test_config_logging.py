import json
import logging
from pathlib import Path

import pytest

from data_ml_toolkit.config import find_repo_root, load_yaml, resolve_path
from data_ml_toolkit.logging import get_logger, log_event


def test_load_yaml_reads_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("paths:\n  raw: data/raw\n", encoding="utf-8")
    assert load_yaml(path) == {"paths": {"raw": "data/raw"}}


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "just a string\n"])
def test_load_yaml_rejects_empty_or_non_mapping(tmp_path: Path, text: str) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="not a mapping"):
        load_yaml(path)


def test_load_yaml_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")


def test_find_repo_root_walks_up(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").touch()
    deep = tmp_path / "src" / "pkg"
    deep.mkdir(parents=True)
    assert find_repo_root(deep) == tmp_path
    with pytest.raises(FileNotFoundError):
        find_repo_root(deep, marker="no-such-marker.txt")


def test_resolve_path(tmp_path: Path) -> None:
    assert resolve_path(tmp_path, "data/raw") == tmp_path / "data" / "raw"
    assert resolve_path(tmp_path, "/abs/path") == Path("/abs/path")


def test_log_event_writes_one_json_object(capsys: pytest.CaptureFixture[str]) -> None:
    log_event(get_logger("toolkit-test"), logging.WARNING, "drift", repo="x", count=2)
    record = json.loads(capsys.readouterr().err.strip())
    assert record["event"] == "drift"
    assert record["level"] == "WARNING"
    assert record["repo"] == "x"
    assert record["count"] == 2


def test_get_logger_adds_one_handler() -> None:
    logger = get_logger("toolkit-once")
    assert get_logger("toolkit-once") is logger
    assert len(logger.handlers) == 1
