"""Contracts for Bank FDS UI configuration helpers."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from kaggle_bank_fds.src.ui.app_config import (
    AppConfig,
    DEFAULT_RULESET_VERSION,
    DEFAULT_SOURCE_NAME,
    prepare_db_parent_directory,
    resolve_bank_fds_db_path,
)


def test_app_config_defaults_and_immutability(tmp_path):
    config = AppConfig(db_path=tmp_path / "bank.sqlite3")
    assert config.ruleset_version == DEFAULT_RULESET_VERSION
    assert config.source_name == DEFAULT_SOURCE_NAME
    assert (config.max_upload_rows, config.max_upload_bytes) == (10_000, 20 * 1024 * 1024)
    assert (config.preview_rows, config.max_evidence_rows, config.alert_list_limit) == (20, 200, 100)
    with pytest.raises(FrozenInstanceError):
        config.preview_rows = 1


@pytest.mark.parametrize("field", [
    "max_upload_rows", "max_upload_bytes", "preview_rows", "max_evidence_rows",
    "alert_list_limit",
])
@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_app_config_rejects_non_integer_limits(tmp_path, field, value):
    with pytest.raises(TypeError):
        AppConfig(db_path=tmp_path / "db.sqlite3", **{field: value})


@pytest.mark.parametrize("field", [
    "max_upload_rows", "max_upload_bytes", "preview_rows", "max_evidence_rows",
    "alert_list_limit",
])
@pytest.mark.parametrize("value", [0, -1])
def test_app_config_rejects_nonpositive_limits(tmp_path, field, value):
    with pytest.raises(ValueError):
        AppConfig(db_path=tmp_path / "db.sqlite3", **{field: value})


def test_app_config_rejects_excessive_alert_limit_and_blank_text(tmp_path):
    with pytest.raises(ValueError, match="1000"):
        AppConfig(db_path=tmp_path / "db.sqlite3", alert_list_limit=1001)
    with pytest.raises(ValueError):
        AppConfig(db_path=tmp_path / "db.sqlite3", source_name=" ")
    with pytest.raises(ValueError):
        AppConfig(db_path=tmp_path / "db.sqlite3", ruleset_version="")


def test_db_path_environment_precedence_expansion_and_relative_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = resolve_bank_fds_db_path(env={"BANK_FDS_DB_PATH": "data/custom.db"})
    assert relative == (tmp_path / "data/custom.db").resolve()
    expanded = resolve_bank_fds_db_path(
        env={"BANK_FDS_DB_PATH": "~/custom.db"}, home=tmp_path
    )
    assert expanded == Path("~/custom.db").expanduser().resolve()


def test_db_path_home_fallback_and_blank_environment(tmp_path):
    assert resolve_bank_fds_db_path(env={}, home=tmp_path) == (
        tmp_path / ".fds_model/bank_fds.sqlite3"
    ).resolve()
    with pytest.raises(ValueError, match="BANK_FDS_DB_PATH"):
        resolve_bank_fds_db_path(env={"BANK_FDS_DB_PATH": "  "}, home=tmp_path)


def test_prepare_parent_is_explicit_and_does_not_create_database(tmp_path):
    path = tmp_path / "nested" / "bank.sqlite3"
    assert not path.parent.exists()
    prepare_db_parent_directory(path)
    assert path.parent.is_dir() and not path.exists()


def test_directory_cannot_be_used_as_database_path(tmp_path):
    with pytest.raises(ValueError, match="directory"):
        AppConfig(db_path=tmp_path)
    with pytest.raises(ValueError, match="directory"):
        prepare_db_parent_directory(tmp_path)
