"""Atomic save contracts for the bank FDS SQLite repository."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import sqlite3

import pytest

from kaggle_bank_fds.src.persistence.fds_result_repository import (
    FdsResultRepository,
    RepositoryClosedError,
)
from kaggle_bank_fds.src.persistence.persistence_models import (
    AnalysisPersistenceArtifact,
    RuleExecutionErrorSnapshot,
    RuleFindingSnapshot,
    TransactionSnapshot,
)
from kaggle_bank_fds.src.persistence.sqlite_schema import SchemaValidationError


CREATED_AT = datetime(2026, 8, 4, 1, 2, 3, tzinfo=timezone.utc)


def _transaction(transaction_id="tx-1", position=0, source_row_id=0, **changes):
    values = dict(
        canonical_transaction_id=transaction_id,
        source_row_id=source_row_id,
        source_position=position,
        step=position + 1,
        transaction_datetime=CREATED_AT,
        action_type="TRANSFER",
        amount=70_000.0,
        actor_account="actor",
        target_account="target",
        counterparty_name="counterparty",
        balance_before=200_000.0,
        balance_after=130_000.0,
        target_balance_before=0.0,
        target_balance_after=70_000.0,
        description="description",
        bank_name="PaySim",
        source_format="PAYSIM",
    )
    values.update(changes)
    return TransactionSnapshot(**values)


def _finding(rule_id, order, score=0, evidence=(), **changes):
    values = dict(
        rule_id=rule_id,
        rule_name=rule_id.replace("_", " ").title(),
        risk_type="risk",
        triggered=score > 0,
        rule_score=score,
        reason="reason",
        execution_order=order,
        evidence_transaction_ids=evidence,
    )
    values.update(changes)
    return RuleFindingSnapshot(**values)


def _artifact(
    run_id="run-1", transactions=None, findings=None, errors=(), score=0.0, **changes
):
    if transactions is None:
        transactions = (_transaction(),)
    if findings is None:
        findings = tuple(_finding(f"rule-{index}", index) for index in range(5))
    values = dict(
        analysis_run_id=run_id,
        source_name="sample.csv",
        ruleset_version="rules-v1",
        created_at=CREATED_AT,
        input_row_count=len(transactions),
        fraud_score=score,
        transactions=transactions,
        rule_findings=findings,
        rule_errors=errors,
    )
    values.update(changes)
    return AnalysisPersistenceArtifact(**values)


@pytest.fixture
def connection():
    value = sqlite3.connect(":memory:")
    yield value
    value.close()


@pytest.fixture
def repository(connection):
    value = FdsResultRepository(connection, alert_id_factory=lambda: "alert-fixed")
    value.initialize_schema()
    return value


def _counts(connection):
    return tuple(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in (
        "analysis_runs", "transaction_snapshots", "alerts", "rule_findings",
        "finding_evidence", "rule_execution_errors",
    ))


def test_save_requires_explicit_schema_initialization(connection):
    repository = FdsResultRepository(connection)
    with pytest.raises(SchemaValidationError):
        repository.save_analysis(_artifact())
    assert not connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_runs'"
    ).fetchall()


def test_external_connection_lifecycle_and_closed_repository(connection):
    repository = FdsResultRepository(connection)
    repository.initialize_schema()
    repository.close()
    repository.close()
    assert connection.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(RepositoryClosedError):
        repository.save_analysis(_artifact())
    with pytest.raises(RepositoryClosedError):
        repository.initialize_schema()


def test_already_closed_external_connection_is_reported():
    connection = sqlite3.connect(":memory:")
    repository = FdsResultRepository(connection)
    connection.close()
    with pytest.raises(RepositoryClosedError, match="connection"):
        repository.initialize_schema()


def test_clean_run_mapping_and_return_value(repository, connection):
    artifact = _artifact()
    assert repository.save_analysis(artifact) == "run-1"
    assert _counts(connection) == (1, 1, 0, 5, 0, 0)
    assert connection.execute(
        """SELECT source_name, ruleset_version, created_at, input_row_count,
                  fraud_score, transaction_count, finding_count, error_count
           FROM analysis_runs"""
    ).fetchone() == ("sample.csv", "rules-v1", CREATED_AT.isoformat(), 1, 0.0, 1, 5, 0)


def test_exact_overlap_preserves_scores_and_evidence_order(repository, connection):
    transactions = tuple(_transaction(f"tx-{index}", index) for index in range(3))
    findings = (
        _finding("rapid_repeated_transfer", 0, 30, ("tx-0", "tx-1", "tx-2")),
        _finding("split_transaction", 1, 35, ("tx-2", "tx-0", "tx-1")),
    )
    repository.save_analysis(_artifact(transactions=transactions, findings=findings, score=.35))
    assert connection.execute(
        "SELECT status, risk_level, fraud_score FROM alerts"
    ).fetchone() == ("OPEN", None, .35)
    assert connection.execute(
        "SELECT rule_id, rule_score FROM rule_findings ORDER BY execution_order"
    ).fetchall() == [("rapid_repeated_transfer", 30), ("split_transaction", 35)]
    rows = connection.execute(
        """SELECT f.rule_id, t.canonical_transaction_id
           FROM finding_evidence e JOIN rule_findings f USING (finding_id)
           JOIN transaction_snapshots t USING (snapshot_id)
           ORDER BY f.execution_order, e.evidence_order"""
    ).fetchall()
    assert rows == [
        ("rapid_repeated_transfer", "tx-0"), ("rapid_repeated_transfer", "tx-1"),
        ("rapid_repeated_transfer", "tx-2"), ("split_transaction", "tx-2"),
        ("split_transaction", "tx-0"), ("split_transaction", "tx-1"),
    ]


def test_partial_overlap_is_stored_without_score_recalculation(repository, connection):
    transactions = tuple(_transaction(f"tx-{index}", index) for index in range(4))
    findings = (
        _finding("rapid_repeated_transfer", 0, 30, ("tx-0", "tx-1", "tx-2")),
        _finding("split_transaction", 1, 35, ("tx-0", "tx-1", "tx-2", "tx-3")),
    )
    repository.save_analysis(_artifact(transactions=transactions, findings=findings, score=.65))
    assert connection.execute("SELECT fraud_score FROM analysis_runs").fetchone()[0] == .65
    assert connection.execute("SELECT count(*) FROM finding_evidence").fetchone()[0] == 7


def test_error_row_is_preserved(repository, connection):
    finding = _finding("success", 0)
    error = RuleExecutionErrorSnapshot("failed", "Failed", "ValueError", "safe message", 1)
    repository.save_analysis(_artifact(findings=(finding,), errors=(error,)))
    assert connection.execute(
        "SELECT rule_id, rule_name, error_type, message, execution_order FROM rule_execution_errors"
    ).fetchone() == ("failed", "Failed", "ValueError", "safe message", 1)
    assert connection.execute("SELECT count(*) FROM alerts").fetchone()[0] == 0


def test_source_row_ids_preserve_int_string_and_none(repository, connection):
    transactions = (
        _transaction("tx-int", 0, 10),
        _transaction("tx-str", 1, "10"),
        _transaction("tx-none", 2, None),
    )
    repository.save_analysis(_artifact(transactions=transactions))
    assert connection.execute(
        """SELECT source_row_id_type, source_row_id_text
           FROM transaction_snapshots ORDER BY source_position"""
    ).fetchall() == [("int", "10"), ("str", "10"), ("none", None)]


def test_naive_transaction_datetime_fails_before_any_insert(repository, connection):
    artifact = _artifact(transactions=(_transaction(transaction_datetime=datetime(2026, 8, 4)),))
    with pytest.raises(ValueError, match="timezone-aware"):
        repository.save_analysis(artifact)
    assert _counts(connection) == (0, 0, 0, 0, 0, 0)


def test_sql_trigger_failure_rolls_back_only_new_run(repository, connection):
    repository.save_analysis(_artifact("existing"))
    connection.execute(
        """CREATE TRIGGER abort_findings BEFORE INSERT ON rule_findings
           BEGIN SELECT RAISE(ABORT, 'controlled failure'); END"""
    )
    with pytest.raises(sqlite3.IntegrityError, match="controlled failure"):
        repository.save_analysis(_artifact("failed"))
    assert connection.execute(
        "SELECT analysis_run_id FROM analysis_runs"
    ).fetchall() == [("existing",)]
    connection.execute("DROP TRIGGER abort_findings")
    assert repository.save_analysis(_artifact("reused")) == "reused"


def test_save_inside_caller_transaction_uses_savepoint(repository, connection):
    connection.execute("CREATE TABLE caller_data (value TEXT)")
    connection.execute("INSERT INTO caller_data VALUES ('pending')")
    repository.save_analysis(_artifact())
    assert connection.in_transaction
    connection.rollback()
    assert connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 0
    assert connection.execute("SELECT count(*) FROM caller_data").fetchone()[0] == 0


def test_duplicate_run_rolls_back_second_attempt_and_new_run_can_reuse_ids(
    repository, connection
):
    artifact = _artifact()
    repository.save_analysis(artifact)
    original = _counts(connection)
    with pytest.raises(sqlite3.IntegrityError):
        repository.save_analysis(artifact)
    assert _counts(connection) == original
    repository.save_analysis(replace(artifact, analysis_run_id="run-2"))
    assert connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 2


@pytest.mark.parametrize("score,triggered,expected", [
    (0.0, False, 0), (.1, False, 1), (0.0, True, 1), (.1, True, 1),
])
def test_alert_uses_score_or_triggered_policy(repository, connection, score, triggered, expected):
    finding = _finding("rule", 0, 20 if triggered else 0, ("tx-1",) if triggered else ())
    repository.save_analysis(_artifact(findings=(finding,), score=score))
    assert connection.execute("SELECT count(*) FROM alerts").fetchone()[0] == expected


def test_parameter_binding_preserves_sql_like_values(repository, connection):
    payload = "x'); DROP TABLE analysis_runs; --"
    transaction = _transaction(actor_account=payload)
    finding = _finding("rule", 0, 20, ("tx-1",), reason=payload)
    artifact = _artifact(
        source_name=payload, transactions=(transaction,), findings=(finding,), score=.2
    )
    repository.save_analysis(artifact)
    assert connection.execute("SELECT source_name FROM analysis_runs").fetchone()[0] == payload
    assert connection.execute("SELECT actor_account FROM transaction_snapshots").fetchone()[0] == payload
    assert connection.execute("SELECT reason FROM rule_findings").fetchone()[0] == payload
    assert connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 1


def test_save_does_not_mutate_artifact(repository):
    artifact = _artifact()
    before = deepcopy(artifact)
    repository.save_analysis(artifact)
    assert artifact == before


def test_owned_file_database_round_trip_and_close(tmp_path):
    path = tmp_path / "result.sqlite3"
    repository = FdsResultRepository.from_path(path, alert_id_factory=lambda: "alert")
    repository.initialize_schema()
    repository.save_analysis(_artifact())
    owned_connection = repository._connection
    repository.close()
    with pytest.raises(sqlite3.ProgrammingError):
        owned_connection.execute("SELECT 1")
    reopened = sqlite3.connect(path)
    second = FdsResultRepository(reopened)
    assert reopened.execute("PRAGMA user_version").fetchone()[0] == 1
    assert reopened.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert reopened.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert reopened.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 1
    assert reopened.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    second.close()
    reopened.close()
