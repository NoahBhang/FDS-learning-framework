"""End-to-end contracts for the Streamlit new-analysis workflow."""

from __future__ import annotations

import importlib
from pathlib import Path
import sqlite3

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from streamlit.testing.v1 import AppTest

from kaggle_bank_fds.src.persistence import FdsResultRepository
from kaggle_bank_fds.src.ui.app_config import AppConfig
from kaggle_bank_fds.src.ui.csv_preflight import parse_and_validate_paysim_csv


APP_PATH = Path(__file__).parents[1] / "src" / "ui" / "streamlit_app.py"
COLUMNS = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
)


def _row(step=1, action="PAYMENT", amount=100.0, actor="A", target="M",
         balance=500_000.0):
    return {
        "step": step, "type": action, "amount": amount, "nameOrig": actor,
        "oldbalanceOrg": balance, "newbalanceOrig": balance - amount,
        "nameDest": target, "oldbalanceDest": 0.0, "newbalanceDest": amount,
    }


def _csv(rows) -> bytes:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.to_csv(index=False).encode()


def _parsed(rows):
    content = _csv(rows)
    return parse_and_validate_paysim_csv(
        content, filename="transactions.csv", file_size_bytes=len(content)
    )


def _app(monkeypatch, db_path: Path) -> AppTest:
    monkeypatch.setenv("BANK_FDS_DB_PATH", str(db_path))
    return AppTest.from_file(str(APP_PATH), default_timeout=20).run()


def _counts(db_path: Path) -> tuple[int, int]:
    with sqlite3.connect(db_path) as connection:
        runs = connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0]
        alerts = connection.execute("SELECT count(*) FROM alerts").fetchone()[0]
    return runs, alerts


def _upload_and_run(at: AppTest, rows) -> AppTest:
    content = _csv(rows)
    at.file_uploader[0].upload("transactions.csv", content, "text/csv").run()
    assert at.button[0].label == "분석 실행" and not at.button[0].disabled
    return at.button[0].click().run()


def test_clean_analysis_persists_once_and_restores_summary_on_rerun(monkeypatch, tmp_path):
    db_path = tmp_path / "operations.sqlite3"
    at = _upload_and_run(_app(monkeypatch, db_path), [_row()])

    assert not at.exception
    metrics = {(value.label, value.value) for value in at.metric}
    assert ("규칙 기반 위험점수", "0/100") in metrics
    assert ("거래 수", "1") in metrics
    assert ("탐지 Rule 수", "0") in metrics
    assert ("오류 Rule 수", "0") in metrics
    assert len([value for value in at.expander if "Rule · 0/100" in value.label]) == 5
    assert _counts(db_path) == (1, 0)
    run_id = at.session_state["bank_fds_last_run_id"]
    assert run_id and at.session_state["bank_fds_last_alert_id"] is None

    at.run()
    assert _counts(db_path) == (1, 0)
    assert at.button[0].label == "같은 파일 다시 분석"
    assert any("상세 clean finding 재조회" in value.value for value in at.info)
    assert any(run_id in str(value.value) for value in at.get("markdown"))


def test_exact_overlap_renders_policy_score_rules_evidence_and_restores(monkeypatch, tmp_path):
    db_path = tmp_path / "operations.sqlite3"
    rows = [
        _row(1, "TRANSFER", 70_000, target="B"),
        _row(2, "TRANSFER", 70_000, target="B"),
        _row(3, "TRANSFER", 70_000, target="B"),
    ]
    at = _upload_and_run(_app(monkeypatch, db_path), rows)

    assert not at.exception and _counts(db_path) == (1, 1)
    metrics = {(value.label, value.value) for value in at.metric}
    assert ("규칙 기반 위험점수", "35/100") in metrics
    labels = [value.label for value in at.expander]
    assert any("Rapid Repeated Transfer Rule · 30/100" in value for value in labels)
    assert any("Split Transaction Rule · 35/100" in value for value in labels)
    basic_columns = {"순서", "거래 ID", "단계", "거래 유형", "금액", "출발 계좌", "도착 계좌"}
    evidence_tables = [
        value.value for value in at.dataframe
        if set(value.value.columns) == basic_columns
    ]
    assert len(evidence_tables) == 4
    assert all(len(value) == 3 for value in evidence_tables)
    assert all(list(value["순서"]) == [0, 1, 2] for value in evidence_tables)
    alert_id = at.session_state["bank_fds_last_alert_id"]
    assert alert_id

    at.run()
    assert _counts(db_path) == (1, 1)
    assert at.session_state["bank_fds_last_alert_id"] == alert_id
    assert any("35/100" == value.value for value in at.metric)
    assert len([
        value for value in at.dataframe if set(value.value.columns) == basic_columns
    ]) == 4


def test_only_explicit_reanalysis_creates_second_run_and_alert(monkeypatch, tmp_path):
    db_path = tmp_path / "operations.sqlite3"
    rows = [
        _row(1, "TRANSFER", 70_000, target="B"),
        _row(2, "TRANSFER", 70_000, target="B"),
        _row(3, "TRANSFER", 70_000, target="B"),
    ]
    at = _upload_and_run(_app(monkeypatch, db_path), rows)
    at.run()
    assert _counts(db_path) == (1, 1)
    assert at.button[0].label == "같은 파일 다시 분석"
    at.button[0].click().run()
    assert _counts(db_path) == (2, 2)


def test_upload_change_clears_previous_result_without_new_save(monkeypatch, tmp_path):
    db_path = tmp_path / "operations.sqlite3"
    at = _upload_and_run(_app(monkeypatch, db_path), [_row()])
    different = _csv([_row(step=2, amount=200)])
    at.file_uploader[0].upload("different.csv", different, "text/csv").run()
    assert _counts(db_path) == (1, 0)
    assert at.session_state["bank_fds_last_run_id"] is None
    assert not [metric for metric in at.metric if metric.label == "규칙 기반 위험점수"]
    assert at.button[0].label == "분석 실행"


def test_partial_overlap_score_and_input_dataframe_are_preserved(tmp_path):
    module = importlib.import_module("kaggle_bank_fds.src.ui.streamlit_app")
    parsed = _parsed([
        _row(1, "TRANSFER", 70_000, target="B"),
        _row(2, "TRANSFER", 70_000, target="B"),
        _row(3, "TRANSFER", 70_000, target="B"),
        _row(4, "TRANSFER", 10_000, target="C"),
    ])
    parsed.dataframe.index = pd.Index(["dup", "dup", "x", "x"])
    before = parsed.dataframe.copy(deep=True)
    config = AppConfig(db_path=tmp_path / "operations.sqlite3")
    result = module._run_analysis(parsed, config)

    assert result.artifact.fraud_score == result.prediction["fraud_score"] == .65
    findings = {value.rule_id: value for value in result.alert.findings}
    assert len(findings["rapid_repeated_transfer"].evidence) == 3
    assert len(findings["split_transaction"].evidence) == 4
    assert [value.evidence_order for value in findings["split_transaction"].evidence] == [
        0, 1, 2, 3,
    ]
    assert_frame_equal(parsed.dataframe, before, check_dtype=True)
    assert _counts(config.db_path) == (1, 1)


def test_orchestration_calls_service_once_closes_repository_and_propagates_failure(
    monkeypatch, tmp_path
):
    module = importlib.import_module("kaggle_bank_fds.src.ui.streamlit_app")
    parsed = _parsed([_row()])
    config = AppConfig(db_path=tmp_path / "operations.sqlite3")
    calls = 0

    def fail_once(self, dataframe):
        nonlocal calls
        calls += 1
        assert dataframe is parsed.dataframe
        raise sqlite3.OperationalError("controlled")

    monkeypatch.setattr(module.FdsAnalysisService, "analyze_and_persist", fail_once)
    with pytest.raises(sqlite3.OperationalError, match="controlled"):
        module._run_analysis(parsed, config)
    assert calls == 1
    with FdsResultRepository.from_path(config.db_path) as repository:
        repository.initialize_schema()
        assert repository.list_alerts() == ()


def test_app_analysis_failure_is_redacted_and_does_not_store_success(monkeypatch, tmp_path):
    module = importlib.import_module("kaggle_bank_fds.src.services.fds_analysis_service")

    def fail(self, dataframe):
        raise RuntimeError("secret-account-C123 traceback")

    monkeypatch.setattr(module.FdsAnalysisService, "analyze_and_persist", fail)
    db_path = tmp_path / "operations.sqlite3"
    at = _upload_and_run(_app(monkeypatch, db_path), [_row()])
    errors = " ".join(str(value.value) for value in at.error)
    assert "분석을 완료하지 못했습니다" in errors
    assert "secret-account" not in errors and "traceback" not in errors.lower()
    assert not at.exception
    assert at.session_state["bank_fds_last_run_id"] is None
    assert _counts(db_path) == (0, 0)


def test_evidence_rendering_is_truncated_with_synthetic_caption():
    script = """
from kaggle_bank_fds.src.persistence import EvidenceRecord
from kaggle_bank_fds.src.ui.streamlit_app import _render_evidence
rows = tuple(
    EvidenceRecord(
        f"tx-{i}", i, i, i + 1, None, "TRANSFER", 1000.0,
        "actor", "target", None, None, None, None, None, None,
        "PaySim", "PAYSIM", i,
    )
    for i in range(205)
)
_render_evidence(rows, limit=200)
"""
    at = AppTest.from_string(script).run()
    assert not at.exception and len(at.dataframe[0].value) == 200
    assert list(at.dataframe[0].value["순서"][:3]) == [0, 1, 2]
    caption = " ".join(str(value.value) for value in at.caption)
    assert "전체 205건 중 처음 200건" in caption
    assert "PaySim 합성 거래 데이터" in caption


def test_rule_error_rendering_has_safe_fields_and_no_traceback():
    script = """
from kaggle_bank_fds.src.ui.presenters import ErrorView
from kaggle_bank_fds.src.ui.streamlit_app import _render_errors
_render_errors((ErrorView(2, "2", "rule-id", "Rule Name", "RuntimeError", "safe message"),))
"""
    at = AppTest.from_string(script).run()
    assert not at.exception
    assert any("일부 Rule 실행 중 오류" in str(value.value) for value in at.warning)
    text = " ".join(str(value.value) for value in at.get("markdown"))
    assert "Rule Name (rule-id)" in text
    assert "RuntimeError" in text and "safe message" in text
    assert "traceback" not in text.lower()
