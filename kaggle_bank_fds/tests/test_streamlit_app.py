"""Streamlit shell, repository lifecycle, and empty-state contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
from pathlib import Path
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

from kaggle_bank_fds.src.persistence import (
    AnalysisPersistenceArtifact,
    FdsResultRepository,
    RepositoryClosedError,
    RuleFindingSnapshot,
    TransactionSnapshot,
)
from kaggle_bank_fds.src.ui.csv_preflight import (
    UnsupportedCsvFileError,
    parse_and_validate_paysim_csv,
)


APP_PATH = Path(__file__).parents[1] / "src" / "ui" / "streamlit_app.py"
REQUIRED_COLUMNS = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
)


def _csv(*, rows: int = 1, missing: str | None = None) -> bytes:
    columns = [value for value in REQUIRED_COLUMNS if value != missing]
    header = ",".join(columns)
    row = {
        "step": "1", "type": "TRANSFER", "amount": "1000", "nameOrig": "C1",
        "oldbalanceOrg": "2000", "newbalanceOrig": "1000", "nameDest": "C2",
        "oldbalanceDest": "0", "newbalanceDest": "1000",
    }
    body = ",".join(row[column] for column in columns)
    return (header + "\n" + "\n".join([body] * rows) + "\n").encode()


def _run(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> AppTest:
    monkeypatch.setenv("BANK_FDS_DB_PATH", str(db_path))
    return AppTest.from_file(str(APP_PATH), default_timeout=15).run()


def _texts(elements) -> list[str]:
    return [str(element.value) for element in elements]


def _transaction() -> TransactionSnapshot:
    return TransactionSnapshot(
        canonical_transaction_id="tx-1", source_row_id=0, source_position=0,
        step=1, transaction_datetime=None, action_type="TRANSFER", amount=70_000.0,
        actor_account="actor", target_account="target", counterparty_name=None,
        balance_before=100_000.0, balance_after=30_000.0,
        target_balance_before=0.0, target_balance_after=70_000.0,
        description=None, bank_name="PaySim", source_format="PAYSIM",
    )


def _save_alert(db_path: Path, run_id: str, alert_id: str) -> None:
    artifact = AnalysisPersistenceArtifact(
        analysis_run_id=run_id,
        source_name="fixture.csv",
        ruleset_version="rules-v1",
        created_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
        input_row_count=1,
        fraud_score=.35,
        transactions=(_transaction(),),
        rule_findings=(RuleFindingSnapshot(
            "rule", "Rule", "risk", True, 35, "reason", 0, ("tx-1",)
        ),),
    )
    with FdsResultRepository.from_path(
        db_path, alert_id_factory=lambda: alert_id
    ) as repository:
        repository.initialize_schema()
        repository.save_analysis(artifact)


def test_module_import_has_no_app_or_database_side_effect(monkeypatch, tmp_path):
    db_path = tmp_path / "not-created.sqlite3"
    monkeypatch.setenv("BANK_FDS_DB_PATH", str(db_path))
    module = importlib.import_module("kaggle_bank_fds.src.ui.streamlit_app")
    assert callable(module.main)
    assert not db_path.exists()


def test_shell_renders_page_tabs_sidebar_and_empty_states(monkeypatch, tmp_path):
    home = tmp_path / "unused-home"
    monkeypatch.setenv("HOME", str(home))
    at = _run(monkeypatch, tmp_path / "data" / "operations.sqlite3")

    assert not at.exception
    assert _texts(at.title) == ["은행 거래 이상징후 탐지 시스템"]
    assert [tab.label for tab in at.tabs] == ["새 분석", "Alert 이력"]
    assert "DB 준비 완료" in _texts(at.sidebar.success)
    sidebar_text = " ".join(_texts(at.sidebar.get("markdown")))
    assert "Explainable Rule-Based FDS" in sidebar_text
    assert "PaySim synthetic" in sidebar_text
    assert "SQLite schema v1" in sidebar_text
    assert any("업로드해 주세요" in value for value in _texts(at.info))
    assert any("저장된 Alert가 없습니다" in value for value in _texts(at.info))
    assert at.button[0].label == "분석 실행" and at.button[0].disabled
    assert not (home / ".fds_model" / "bank_fds.sqlite3").exists()


def test_run_initializes_schema_and_allows_reopen_without_lock(monkeypatch, tmp_path):
    db_path = tmp_path / "operations.sqlite3"
    at = _run(monkeypatch, db_path)
    assert not at.exception and db_path.is_file()
    with FdsResultRepository.from_path(db_path) as repository:
        assert repository.list_alerts() == ()
    at.run()
    with sqlite3.connect(db_path, timeout=.1) as connection:
        connection.execute("BEGIN EXCLUSIVE")
        connection.rollback()


@pytest.mark.parametrize(
    ("filename", "payload", "expected"),
    [
        ("transactions.csv", _csv(missing="nameDest"), "필수 CSV 컬럼이 없습니다"),
        ("transactions.csv", _csv(rows=10_001), "CSV 행 수가 허용된 한도"),
    ],
)
def test_invalid_upload_states(monkeypatch, tmp_path, filename, payload, expected):
    at = _run(monkeypatch, tmp_path / "operations.sqlite3")
    at.file_uploader[0].upload(filename, payload, "text/csv").run()
    assert not at.exception
    assert any(expected in value for value in _texts(at.error))
    assert at.button[0].disabled


def test_uploader_and_preflight_reject_invalid_extension(monkeypatch, tmp_path):
    at = _run(monkeypatch, tmp_path / "operations.sqlite3")
    assert at.file_uploader[0].allowed_type == [".csv"]
    payload = _csv()
    with pytest.raises(UnsupportedCsvFileError, match="CSV 확장자 파일만"):
        parse_and_validate_paysim_csv(
            payload,
            filename="transactions.txt",
            file_size_bytes=len(payload),
        )


def test_valid_upload_shows_metadata_metrics_preview_and_enabled_action(monkeypatch, tmp_path):
    at = _run(monkeypatch, tmp_path / "operations.sqlite3")
    payload = _csv(rows=2)
    at.file_uploader[0].upload("transactions.csv", payload, "text/csv").run()

    assert not at.exception
    markdown = " ".join(_texts(at.get("markdown")))
    assert "파일명: transactions.csv" in markdown
    assert f"파일 크기: {len(payload):,} bytes" in markdown
    assert [(metric.label, metric.value) for metric in at.metric] == [
        ("거래 행 수", "2"), ("컬럼 수", "9")
    ]
    assert len(at.dataframe) == 1
    assert "CSV 사전 검증이 완료되었습니다." in _texts(at.success)
    assert not at.button[0].disabled


def test_saved_alert_list_and_selection_survive_rerun(monkeypatch, tmp_path):
    db_path = tmp_path / "operations.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _save_alert(db_path, "run-1", "alert-1")
    _save_alert(db_path, "run-2", "alert-2")

    at = _run(monkeypatch, db_path)
    assert not at.exception
    alert_table = next(
        value.value for value in at.dataframe if "Alert ID" in value.value.columns
    )
    assert set(alert_table["Alert ID"]) == {"alert-1", "alert-2"}
    at.selectbox[0].select("alert-1").run()
    assert at.session_state["bank_fds_selected_alert_id"] == "alert-1"
    at.run()
    assert at.selectbox[0].value == "alert-1"
    assert any("분석 Run ID: run-1" in value for value in _texts(at.get("markdown")))


def test_operation_repository_is_closed_after_context(tmp_path):
    module = importlib.import_module("kaggle_bank_fds.src.ui.streamlit_app")
    config = module.AppConfig(db_path=tmp_path / "operations.sqlite3")
    with module._open_repository(config) as repository:
        assert repository.list_alerts() == ()
    with pytest.raises(RepositoryClosedError):
        repository.list_alerts()


def test_project_tree_contains_no_database_files():
    project = APP_PATH.parents[3]
    database_suffixes = {".db", ".sqlite", ".sqlite3"}
    assert not [path for path in project.rglob("*") if path.suffix in database_suffixes]
