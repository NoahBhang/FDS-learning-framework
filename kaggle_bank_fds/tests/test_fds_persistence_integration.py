"""Real Rule end-to-end persistence integration contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

import pandas as pd

from kaggle_bank_fds.src.persistence.fds_result_repository import FdsResultRepository
from kaggle_bank_fds.src.rules.full_balance_transfer_rule import FullBalanceTransferRule
from kaggle_bank_fds.src.rules.rounded_amount_rule import RoundedAmountRule
from kaggle_bank_fds.src.services.fds_analysis_service import FdsAnalysisService


CREATED = datetime(2026, 8, 4, tzinfo=timezone.utc)


def _row(step, amount, target, action="TRANSFER", actor="A", balance=500_000.0):
    return {
        "step": step, "type": action, "amount": amount, "nameOrig": actor,
        "oldbalanceOrg": balance, "newbalanceOrig": balance - amount,
        "nameDest": target, "oldbalanceDest": 0.0, "newbalanceDest": amount,
    }


def _service(connection, alert_id="alert"):
    repository = FdsResultRepository(connection, alert_id_factory=lambda: alert_id)
    repository.initialize_schema()
    return FdsAnalysisService(
        repository, source_name="integration.csv",
        ruleset_version="bank-fds-default-v1+69106de",
    ), repository


def test_clean_default_end_to_end():
    connection = sqlite3.connect(":memory:"); service, _ = _service(connection)
    result = service.analyze_and_persist(
        pd.DataFrame([_row(1, 100, "M", "PAYMENT")]),
        analysis_run_id="clean", created_at=CREATED,
    )
    assert result.prediction["fraud_score"] == result.artifact.fraud_score == 0.0
    assert result.analysis_run.fraud_score == 0.0 and result.alert is None
    assert len(result.artifact.rule_findings) == 5
    assert tuple(connection.execute(
        f"SELECT count(*) FROM {table}"
    ).fetchone()[0] for table in (
        "analysis_runs", "transaction_snapshots", "alerts", "rule_findings",
        "finding_evidence", "rule_execution_errors",
    )) == (1, 1, 0, 5, 0, 0)
    connection.close()


def test_exact_overlap_duplicate_raw_index_semantic_consistency():
    connection = sqlite3.connect(":memory:"); service, _ = _service(connection)
    frame = pd.DataFrame([
        _row(1, 70_000, "B"), _row(2, 70_000, "B"), _row(3, 70_000, "B"),
    ], index=["same", "same", "same"])
    result = service.analyze_and_persist(frame, analysis_run_id="exact", created_at=CREATED)
    assert result.prediction["fraud_score"] == result.artifact.fraud_score == .35
    assert result.analysis_run.fraud_score == result.alert.summary.fraud_score == .35
    findings = {value.rule_id: value for value in result.artifact.rule_findings}
    read = {value.rule_id: value for value in result.alert.findings}
    assert findings["rapid_repeated_transfer"].rule_score == 30
    assert findings["split_transaction"].rule_score == 35
    assert findings["rapid_repeated_transfer"].evidence_transaction_ids == findings["split_transaction"].evidence_transaction_ids
    for rule_id in ("rapid_repeated_transfer", "split_transaction"):
        assert findings[rule_id].evidence_transaction_ids == tuple(
            item.canonical_transaction_id for item in read[rule_id].evidence
        )
    connection.close()


def test_partial_overlap_semantic_consistency():
    connection = sqlite3.connect(":memory:"); service, _ = _service(connection)
    frame = pd.DataFrame([
        _row(1, 70_000, "B"), _row(2, 70_000, "B"), _row(3, 70_000, "B"),
        _row(4, 10_000, "C"),
    ])
    result = service.analyze_and_persist(frame, analysis_run_id="partial", created_at=CREATED)
    assert result.prediction["fraud_score"] == result.artifact.fraud_score == .65
    assert result.analysis_run.fraud_score == result.alert.summary.fraud_score == .65
    findings = {value.rule_id: value for value in result.alert.findings}
    assert len(findings["rapid_repeated_transfer"].evidence) == 3
    assert len(findings["split_transaction"].evidence) == 4
    connection.close()


def test_independent_rounded_full_balance_overlap_is_not_discounted():
    connection = sqlite3.connect(":memory:"); service, _ = _service(connection)
    frame = pd.DataFrame([_row(1, 100_000, "B", balance=100_000)])
    result = service.analyze_and_persist(
        frame, rules=[RoundedAmountRule(), FullBalanceTransferRule()],
        analysis_run_id="independent", created_at=CREATED,
    )
    assert result.prediction["fraud_score"] == result.artifact.fraud_score == .4
    assert result.alert.triggered_rule_ids == ("rounded_amount", "full_balance_transfer")
    assert [value.rule_score for value in result.alert.findings] == [20, 20]
    connection.close()


def test_raw_index_is_independent_from_positional_source_row_id():
    connection = sqlite3.connect(":memory:"); service, _ = _service(connection)
    frame = pd.DataFrame([
        _row(1, 70_000, "B"), _row(2, 70_000, "B"), _row(3, 70_000, "B"),
    ], index=[10, "10", None])
    result = service.analyze_and_persist(frame, analysis_run_id="ids", created_at=CREATED)
    rapid = next(value for value in result.alert.findings if value.rule_id == "rapid_repeated_transfer")
    assert [value.source_row_id for value in rapid.evidence] == [0, 1, 2]
    connection.close()


def test_file_db_close_reopen_round_trip(tmp_path):
    path = tmp_path / "integration.sqlite3"
    repository = FdsResultRepository.from_path(path, alert_id_factory=lambda: "file-alert")
    repository.initialize_schema()
    service = FdsAnalysisService(repository, source_name="file.csv", ruleset_version="v1")
    result = service.analyze_and_persist(
        pd.DataFrame([_row(1, 100_000, "B", balance=100_000)]),
        rules=[RoundedAmountRule(), FullBalanceTransferRule()],
        analysis_run_id="file-run", created_at=CREATED,
    )
    repository.close()
    reopened = FdsResultRepository.from_path(path)
    assert reopened.get_analysis_run("file-run") == result.analysis_run
    assert reopened.get_alert_detail("file-alert").triggered_rule_ids == result.alert.triggered_rule_ids
    assert reopened._connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert reopened._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert reopened._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    reopened.close()
