"""Alert history selection, typed detail, and semantic display contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pandas as pd
from streamlit.testing.v1 import AppTest

from kaggle_bank_fds.src.persistence import FdsResultRepository
from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.evidence_item import EvidenceItem
from kaggle_bank_fds.src.rules.rule_result import RuleResult
from kaggle_bank_fds.src.services.fds_analysis_service import FdsAnalysisService


APP_PATH = Path(__file__).parents[1] / "src" / "ui" / "streamlit_app.py"
CREATED = datetime(2026, 8, 4, 6, tzinfo=timezone.utc)
BASIC_EVIDENCE_COLUMNS = {
    "순서", "거래 ID", "단계", "거래 유형", "금액", "출발 계좌", "도착 계좌",
}
DETAIL_EVIDENCE_COLUMNS = {
    "순서", "거래 ID", "Source row ID", "거래 시각", "상대 계좌",
    "출발 잔액(전)", "출발 잔액(후)", "도착 잔액(전)", "도착 잔액(후)",
    "설명", "은행 코드", "Source format",
}


def _row(step=1, amount=100.0, target="M", action="PAYMENT", actor="A"):
    return {
        "step": step, "type": action, "amount": amount, "nameOrig": actor,
        "oldbalanceOrg": 500_000.0, "newbalanceOrig": 500_000.0 - amount,
        "nameDest": target, "oldbalanceDest": 0.0, "newbalanceDest": amount,
    }


def _exact_frame():
    return pd.DataFrame([
        _row(1, 70_000, "B", "TRANSFER"),
        _row(2, 70_000, "B", "TRANSFER"),
        _row(3, 70_000, "B", "TRANSFER"),
    ])


def _partial_frame():
    frame = _exact_frame()
    return pd.concat(
        [frame, pd.DataFrame([_row(4, 10_000, "C", "TRANSFER")])],
        ignore_index=True,
    )


def _persist(db_path, frame, *, run_id, alert_id, created_at, rules=None):
    with FdsResultRepository.from_path(
        db_path, alert_id_factory=lambda: alert_id
    ) as repository:
        repository.initialize_schema()
        service = FdsAnalysisService(
            repository, source_name="fixture.csv", ruleset_version="rules-v1"
        )
        return service.analyze_and_persist(
            frame, rules=rules, analysis_run_id=run_id, created_at=created_at
        )


def _app(monkeypatch, db_path):
    monkeypatch.setenv("BANK_FDS_DB_PATH", str(db_path))
    return AppTest.from_file(str(APP_PATH), default_timeout=20).run()


def _counts(db_path):
    with sqlite3.connect(db_path) as connection:
        return (
            connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0],
            connection.execute("SELECT count(*) FROM alerts").fetchone()[0],
        )


def _table(at, required_column):
    return next(value.value for value in at.dataframe if required_column in value.value.columns)


def test_empty_history_preserves_clear_empty_state(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path / "operations.sqlite3")
    assert not at.exception and not at.selectbox
    assert any("저장된 Alert가 없습니다" in str(value.value) for value in at.info)


def test_exact_alert_auto_selects_and_renders_complete_typed_detail(monkeypatch, tmp_path):
    db_path = tmp_path / "operations.sqlite3"
    saved = _persist(
        db_path, _exact_frame(), run_id="exact-run", alert_id="exact-alert",
        created_at=CREATED,
    )
    at = _app(monkeypatch, db_path)

    assert not at.exception
    table = _table(at, "Alert ID")
    assert list(table.columns) == [
        "분석 시각", "규칙 기반 위험점수", "상태", "위험등급", "탐지 Rule 수",
        "Alert ID", "Analysis Run ID",
    ]
    assert table.iloc[0].to_dict()["규칙 기반 위험점수"] == "35/100"
    assert table.iloc[0].to_dict()["위험등급"] == "미분류"
    assert at.selectbox[0].value == "exact-alert"
    assert at.session_state["bank_fds_selected_alert_id"] == "exact-alert"
    metrics = {(value.label, value.value) for value in at.metric}
    assert ("규칙 기반 위험점수", "35/100") in metrics
    assert ("전체 finding 수", "5") in metrics
    assert ("오류 Rule 수", "0") in metrics
    labels = [value.label for value in at.expander]
    assert any("Rapid Repeated Transfer Rule · 30/100" in value for value in labels)
    assert any("Split Transaction Rule · 35/100" in value for value in labels)
    basic = [value.value for value in at.dataframe if set(value.value.columns) == BASIC_EVIDENCE_COLUMNS]
    detail = [value.value for value in at.dataframe if set(value.value.columns) == DETAIL_EVIDENCE_COLUMNS]
    assert len(basic) == len(detail) == 2
    expected_ids = [value.canonical_transaction_id for value in saved.alert.findings[3].evidence]
    assert list(basic[0]["거래 ID"]) == expected_ids
    assert list(basic[0]["순서"]) == [0, 1, 2]


def test_multiple_alerts_keep_repository_order_and_selected_value_on_rerun(monkeypatch, tmp_path):
    db_path = tmp_path / "operations.sqlite3"
    _persist(db_path, _exact_frame(), run_id="old-run", alert_id="old-alert",
             created_at=CREATED - timedelta(hours=1))
    _persist(db_path, _partial_frame(), run_id="new-run", alert_id="new-alert",
             created_at=CREATED)
    at = _app(monkeypatch, db_path)
    table = _table(at, "Alert ID")
    assert list(table["Alert ID"]) == ["new-alert", "old-alert"]
    assert "65/100" in at.selectbox[0].options[0]
    assert "35/100" in at.selectbox[0].options[1]
    assert at.selectbox[0].value == "new-alert"

    at.selectbox[0].select("old-alert").run()
    assert at.session_state["bank_fds_selected_alert_id"] == "old-alert"
    upload = pd.DataFrame([_row()]).to_csv(index=False).encode()
    at.file_uploader[0].upload("clean.csv", upload, "text/csv").run()
    assert at.session_state["bank_fds_selected_alert_id"] == "old-alert"
    at.run()
    assert at.selectbox[0].value == "old-alert"
    assert _counts(db_path) == (2, 2)


def test_partial_overlap_history_keeps_final_and_individual_scores(monkeypatch, tmp_path):
    db_path = tmp_path / "operations.sqlite3"
    _persist(db_path, _partial_frame(), run_id="partial-run", alert_id="partial-alert",
             created_at=CREATED)
    at = _app(monkeypatch, db_path)
    assert ("규칙 기반 위험점수", "65/100") in {
        (value.label, value.value) for value in at.metric
    }
    labels = [value.label for value in at.expander]
    assert any("Rapid Repeated Transfer Rule · 30/100" in value for value in labels)
    assert any("Split Transaction Rule · 35/100" in value for value in labels)
    basic = [value.value for value in at.dataframe if set(value.value.columns) == BASIC_EVIDENCE_COLUMNS]
    assert sorted(len(value) for value in basic) == [3, 4]


class SuccessRule(BaseRule):
    rule_id = "success"; rule_name = "Success Rule"; description = "Success risk"

    def evaluate(self, transactions):
        row = transactions.iloc[0]
        evidence = EvidenceItem(
            row["transaction_id"], int(row["source_row_id"]), row["actor_account"],
            row["target_account"], row["transaction_datetime"], row["amount"], "safe",
        )
        return RuleResult(self.rule_id, self.rule_name, True, 20, "safe", (evidence,))


class FailureRule(BaseRule):
    rule_id = "failure"; rule_name = "Failure Rule"; description = "Failure risk"

    def evaluate(self, transactions):
        raise RuntimeError("redacted safe message")


def test_history_renders_persisted_rule_error_without_traceback(monkeypatch, tmp_path):
    db_path = tmp_path / "operations.sqlite3"
    _persist(
        db_path, pd.DataFrame([_row()]), run_id="error-run", alert_id="error-alert",
        created_at=CREATED, rules=[SuccessRule(), FailureRule()],
    )
    at = _app(monkeypatch, db_path)
    assert not at.exception
    assert any("일부 Rule 실행 중 오류" in str(value.value) for value in at.warning)
    text = " ".join(str(value.value) for value in at.get("markdown"))
    assert "Failure Rule (failure)" in text
    assert "RuntimeError" in text and "redacted safe message" in text
    assert "traceback" not in text.lower()
    assert _counts(db_path) == (1, 1)


def test_missing_selected_alert_is_cleared_without_crash(monkeypatch, tmp_path):
    db_path = tmp_path / "operations.sqlite3"
    _persist(db_path, _exact_frame(), run_id="run", alert_id="alert", created_at=CREATED)
    monkeypatch.setattr(FdsResultRepository, "get_alert_detail", lambda self, alert_id: None)
    at = _app(monkeypatch, db_path)
    assert not at.exception
    assert at.session_state["bank_fds_selected_alert_id"] is None
    assert any("선택한 Alert를 찾을 수 없습니다" in str(value.value) for value in at.warning)
    assert _counts(db_path) == (1, 1)
