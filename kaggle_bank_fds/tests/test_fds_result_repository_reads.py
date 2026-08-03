"""Typed read and semantic round-trip contracts for FDS persistence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
from kaggle_bank_fds.src.persistence.persistence_read_models import (
    AlertDetail,
    AlertSummary,
    AnalysisRunRecord,
)
from kaggle_bank_fds.src.persistence.sqlite_schema import SchemaValidationError


UTC = timezone.utc
CREATED = datetime(2026, 8, 4, 3, 2, 1, tzinfo=UTC)


def _transaction(transaction_id="tx-1", position=0, source_row_id=0, **changes):
    values = dict(
        canonical_transaction_id=transaction_id, source_row_id=source_row_id,
        source_position=position, step=position + 1, transaction_datetime=CREATED,
        action_type="TRANSFER", amount=70_000.0, actor_account="actor",
        target_account="target", counterparty_name="counterparty",
        balance_before=200_000.0, balance_after=130_000.0,
        target_balance_before=0.0, target_balance_after=70_000.0,
        description="description", bank_name="PaySim", source_format="PAYSIM",
    )
    values.update(changes)
    return TransactionSnapshot(**values)


def _finding(rule_id="rule", order=0, score=20, evidence=("tx-1",)):
    return RuleFindingSnapshot(
        rule_id, rule_id.title(), "risk", score > 0, score, "reason", order, evidence
    )


def _artifact(run_id="run-1", *, transactions=None, findings=None, errors=(), score=.2,
              created_at=CREATED, source_name="source.csv"):
    transactions = transactions or (_transaction(),)
    findings = findings if findings is not None else (_finding(),)
    return AnalysisPersistenceArtifact(
        run_id, source_name, "rules-v1", created_at, len(transactions), score,
        transactions, findings, errors,
    )


@pytest.fixture
def setup():
    connection = sqlite3.connect(":memory:")
    repository = FdsResultRepository(connection, alert_id_factory=lambda: "alert-1")
    repository.initialize_schema()
    yield repository, connection
    connection.close()


def test_get_analysis_run_known_unknown_blank_and_clean(setup):
    repository, _ = setup
    artifact = _artifact(findings=(_finding(score=0, evidence=()),), score=0)
    repository.save_analysis(artifact)
    record = repository.get_analysis_run("run-1")
    assert isinstance(record, AnalysisRunRecord)
    assert record == AnalysisRunRecord(
        "run-1", "source.csv", "rules-v1", CREATED, 1, 0.0, 1, 1, 0
    )
    assert repository.get_analysis_run("unknown") is None
    with pytest.raises(ValueError):
        repository.get_analysis_run("  ")


def test_analysis_run_malformed_datetime_is_rejected(setup):
    repository, connection = setup
    repository.save_analysis(_artifact())
    connection.execute("UPDATE analysis_runs SET created_at='not-a-date'")
    with pytest.raises(SchemaValidationError, match="ISO-8601"):
        repository.get_analysis_run("run-1")


def test_list_alerts_empty_sort_tie_count_limit_and_nullable_risk(setup):
    repository, connection = setup
    assert repository.list_alerts() == ()
    clean = _artifact("clean", findings=(_finding(score=0, evidence=()),), score=0)
    repository.save_analysis(clean)
    repository.save_analysis(_artifact("older", created_at=CREATED - timedelta(days=1)))
    repository._alert_id_factory = lambda: "alert-2"
    findings = (_finding("one", 0), _finding("two", 1))
    repository.save_analysis(_artifact("newer-a", findings=findings))
    repository._alert_id_factory = lambda: "alert-3"
    repository.save_analysis(_artifact("newer-b", findings=findings))
    alerts = repository.list_alerts()
    assert all(isinstance(value, AlertSummary) for value in alerts)
    assert [value.alert_id for value in alerts] == ["alert-3", "alert-2", "alert-1"]
    assert alerts[0].triggered_rule_count == 2
    assert alerts[0].risk_level is None and alerts[0].status == "OPEN"
    assert len(repository.list_alerts(limit=1)) == 1
    assert connection.execute("SELECT count(*) FROM alerts WHERE analysis_run_id='clean'").fetchone()[0] == 0


def test_get_alert_by_run_id_known_unknown_blank_and_parameter_binding(setup):
    repository, _ = setup
    repository.save_analysis(_artifact())
    summary = repository.get_alert_by_run_id("run-1")
    assert summary.alert_id == "alert-1" and summary.triggered_rule_count == 1
    assert repository.get_alert_by_run_id("unknown") is None
    assert repository.get_alert_by_run_id("x' OR 1=1; --") is None
    with pytest.raises(ValueError):
        repository.get_alert_by_run_id(" ")


@pytest.mark.parametrize("limit", [0, -1, 1001])
def test_list_alerts_rejects_out_of_range_limit(setup, limit):
    with pytest.raises(ValueError):
        setup[0].list_alerts(limit=limit)


@pytest.mark.parametrize("limit", [True, 1.5, "1"])
def test_list_alerts_rejects_non_integer_limit(setup, limit):
    with pytest.raises(TypeError):
        setup[0].list_alerts(limit=limit)


def test_exact_overlap_detail_semantic_round_trip(setup):
    repository, _ = setup
    transactions = tuple(
        _transaction(f"tx-{index}", index, [10, "10", None][index]) for index in range(3)
    )
    findings = (
        _finding("rapid", 0, 30, ("tx-0", "tx-1", "tx-2")),
        _finding("split", 1, 35, ("tx-0", "tx-1", "tx-2")),
        _finding("clean", 2, 0, ()),
    )
    repository.save_analysis(_artifact(transactions=transactions, findings=findings, score=.35))
    detail = repository.get_alert_detail("alert-1")
    assert isinstance(detail, AlertDetail)
    assert detail.summary.fraud_score == detail.analysis_run.fraud_score == .35
    assert detail.triggered_rule_ids == ("rapid", "split")
    assert [(value.rule_id, value.rule_score) for value in detail.findings] == [
        ("rapid", 30), ("split", 35), ("clean", 0),
    ]
    assert tuple(value.canonical_transaction_id for value in detail.findings[0].evidence) == (
        "tx-0", "tx-1", "tx-2",
    )
    assert [value.source_row_id for value in detail.findings[0].evidence] == [10, "10", None]
    first = detail.findings[0].evidence[0]
    assert (first.action, first.amount, first.actor_account, first.target_account) == (
        "TRANSFER", 70_000.0, "actor", "target",
    )
    assert repository.get_alert_detail("unknown") is None


def test_partial_overlap_preserves_distinct_evidence_order(setup):
    repository, _ = setup
    transactions = tuple(_transaction(f"tx-{index}", index) for index in range(4))
    findings = (
        _finding("rapid", 0, 30, ("tx-2", "tx-0", "tx-1")),
        _finding("split", 1, 35, ("tx-0", "tx-1", "tx-2", "tx-3")),
    )
    repository.save_analysis(_artifact(transactions=transactions, findings=findings, score=.65))
    detail = repository.get_alert_detail("alert-1")
    assert detail.summary.fraud_score == .65
    assert [[item.canonical_transaction_id for item in finding.evidence]
            for finding in detail.findings] == [
        ["tx-2", "tx-0", "tx-1"], ["tx-0", "tx-1", "tx-2", "tx-3"],
    ]


def test_error_detail_order_null_last_and_unknown_alert(setup):
    repository, connection = setup
    errors = (
        RuleExecutionErrorSnapshot("e2", "E2", "ValueError", "two", 2),
        RuleExecutionErrorSnapshot("e1", "E1", "TypeError", "one", 1),
    )
    repository.save_analysis(_artifact(errors=errors))
    connection.execute(
        "UPDATE rule_execution_errors SET execution_order=NULL WHERE rule_id='e2'"
    )
    detail = repository.get_alert_detail("alert-1")
    assert [(value.rule_id, value.error_type, value.message, value.execution_order)
            for value in detail.errors] == [
        ("e1", "TypeError", "one", 1), ("e2", "ValueError", "two", None),
    ]
    assert repository.get_alert_detail("does-not-exist") is None


def test_datetime_round_trip_normalizes_offset_and_preserves_none(setup):
    repository, _ = setup
    offset_time = datetime(2026, 8, 4, 12, 0, tzinfo=timezone(timedelta(hours=9)))
    transactions = (
        _transaction("dated", 0, transaction_datetime=offset_time),
        _transaction("none", 1, transaction_datetime=None),
    )
    finding = _finding(evidence=("dated", "none"))
    repository.save_analysis(_artifact(transactions=transactions, findings=(finding,), created_at=offset_time))
    detail = repository.get_alert_detail("alert-1")
    assert detail.analysis_run.created_at == offset_time.astimezone(UTC)
    assert detail.findings[0].evidence[0].transaction_datetime == offset_time.astimezone(UTC)
    assert detail.findings[0].evidence[1].transaction_datetime is None
    assert repository.get_alert_detail("x' OR 1=1; --") is None


def test_corruption_triggered_without_evidence_is_detected(setup):
    repository, connection = setup
    repository.save_analysis(_artifact())
    connection.execute("DELETE FROM finding_evidence")
    with pytest.raises(SchemaValidationError, match="no evidence"):
        repository.get_alert_detail("alert-1")


def test_corruption_finding_and_error_same_rule_is_detected(setup):
    repository, connection = setup
    repository.save_analysis(_artifact())
    connection.execute(
        """INSERT INTO rule_execution_errors
           (analysis_run_id, rule_id, rule_name, error_type, message, execution_order)
           VALUES ('run-1', 'rule', 'Rule', 'Error', 'message', 2)"""
    )
    with pytest.raises(SchemaValidationError, match="both"):
        repository.get_alert_detail("alert-1")


@pytest.mark.parametrize("column,value,match", [
    ("source_row_id_text", "not-int", "integer source"),
    ("transaction_datetime", "2026-08-04T12:00:00", "timezone-aware"),
    ("transaction_datetime", "not-a-date", "ISO-8601"),
])
def test_corrupt_source_id_or_datetime_is_detected(setup, column, value, match):
    repository, connection = setup
    repository.save_analysis(_artifact())
    connection.execute(f"UPDATE transaction_snapshots SET {column}=?", (value,))
    with pytest.raises(SchemaValidationError, match=match):
        repository.get_alert_detail("alert-1")


def test_alert_detail_has_bounded_select_count(setup):
    repository, connection = setup
    transactions = tuple(_transaction(f"tx-{index}", index) for index in range(100))
    finding = _finding(evidence=tuple(value.canonical_transaction_id for value in transactions))
    repository.save_analysis(_artifact(transactions=transactions, findings=(finding,)))
    traced = []
    connection.set_trace_callback(traced.append)
    detail = repository.get_alert_detail("alert-1")
    connection.set_trace_callback(None)
    selects = [statement for statement in traced if statement.lstrip().upper().startswith("SELECT")]
    assert len(detail.findings[0].evidence) == 100
    # 1 lifecycle probe + 2 schema validation queries + 4 fixed detail queries.
    assert len(selects) <= 7


def test_file_database_semantic_round_trip(tmp_path):
    path = tmp_path / "reads.sqlite3"
    writer = FdsResultRepository.from_path(path, alert_id_factory=lambda: "file-alert")
    writer.initialize_schema()
    writer.save_analysis(_artifact())
    writer.close()
    reader = FdsResultRepository.from_path(path)
    assert reader.get_analysis_run("run-1").source_name == "source.csv"
    assert reader.list_alerts()[0].alert_id == "file-alert"
    assert reader.get_alert_detail("file-alert").findings[0].rule_id == "rule"
    reader.close()


def test_read_apis_respect_repository_lifecycle(setup):
    repository, _ = setup
    repository.close()
    for operation in (
        lambda: repository.get_analysis_run("run"),
        repository.list_alerts,
        lambda: repository.get_alert_by_run_id("run"),
        lambda: repository.get_alert_detail("alert"),
    ):
        with pytest.raises(RepositoryClosedError):
            operation()
