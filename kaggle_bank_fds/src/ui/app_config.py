"""Immutable configuration helpers for the Bank FDS operations UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_RULESET_VERSION = "bank-fds-default-v1+0db90ca"
DEFAULT_SOURCE_NAME = "streamlit-paysim-upload"
DB_PATH_ENV = "BANK_FDS_DB_PATH"


@dataclass(frozen=True, slots=True)
class AppConfig:
    db_path: Path
    ruleset_version: str = DEFAULT_RULESET_VERSION
    source_name: str = DEFAULT_SOURCE_NAME
    max_upload_rows: int = 10_000
    max_upload_bytes: int = 20 * 1024 * 1024
    preview_rows: int = 20
    max_evidence_rows: int = 200
    alert_list_limit: int = 100

    def __post_init__(self) -> None:
        path = Path(self.db_path).expanduser()
        if path.exists() and path.is_dir():
            raise ValueError("db_path must identify a database file, not a directory.")
        if not path.name:
            raise ValueError("db_path must identify a database file.")
        object.__setattr__(self, "db_path", path)
        for field_name in ("ruleset_version", "source_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string.")
            normalized = value.strip()
            if not normalized:
                raise ValueError(f"{field_name} must not be empty.")
            object.__setattr__(self, field_name, normalized)
        for field_name in (
            "max_upload_rows", "max_upload_bytes", "preview_rows",
            "max_evidence_rows", "alert_list_limit",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be a bool-free integer.")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if self.alert_list_limit > 1_000:
            raise ValueError("alert_list_limit must not exceed 1000.")


def resolve_bank_fds_db_path(
    *, env: Mapping[str, str] | None = None, home: Path | None = None
) -> Path:
    """Resolve configuration only; no directory or database is created."""
    environment = os.environ if env is None else env
    configured = environment.get(DB_PATH_ENV)
    if configured is not None:
        if not isinstance(configured, str):
            raise TypeError(f"{DB_PATH_ENV} must be a string.")
        if not configured.strip():
            raise ValueError(f"{DB_PATH_ENV} must not be blank.")
        return Path(configured.strip()).expanduser().resolve()
    base = Path.home() if home is None else Path(home)
    return (base.expanduser() / ".fds_model" / "bank_fds.sqlite3").resolve()


def prepare_db_parent_directory(path: Path) -> None:
    """Create only the parent directory required by a future repository open."""
    normalized = Path(path).expanduser()
    if normalized.exists() and normalized.is_dir():
        raise ValueError("Database path points to an existing directory.")
    normalized.parent.mkdir(parents=True, exist_ok=True)
