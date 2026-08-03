"""Application orchestration contracts for persistent FDS analysis."""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.persistence.fds_result_repository import (
    FdsResultRepository,
    RepositoryClosedError,
)
from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.evidence_item import EvidenceItem
from kaggle_bank_fds.src.rules.rule_result import RuleResult
from kaggle_bank_fds.src.services.fds_analysis_service import (
    AnalysisServiceIntegrityError,
    AnalysisServiceResult,
    FdsAnalysisService,
)


CREATED = datetime(2026, 8, 4, tzinfo=timezone.utc)


def _row(step=1, action="PAYMENT", amount=100.0, actor="A", target="M", balance=500_000.0):
    return {
        "step": step, "type": action, "amount": amount, "nameOrig": actor,
        "oldbalanceOrg": balance, "newbalanceOrig": balance - amount,
        "nameDest": target, "oldbalanceDest": 0.0, "newbalanceDest": amount,
    }


class CountingRule(BaseRule):
    rule_id = "counting"
    rule_name = "Counting Rule"
    description = "Counting risk"

    def __init__(self, *, triggered=False):
        self.calls = 0
        self.triggered = triggered

    def evaluate(self, transactions):
        self.calls += 1
        if not self.triggered:
            return RuleResult(self.rule_id, self.rule_name, False, 0, "clean")
        row = transactions.iloc[0]
        source_row_id = row["source_row_id"]
        if not isinstance(source_row_id, str) and source_row_id is not None:
            source_row_id = int(source_row_id)
        evidence = EvidenceItem(
            row["transaction_id"], source_row_id, row["actor_account"],
            row["target_account"], row["transaction_datetime"], row["amount"], "match",
        )
        return RuleResult(self.rule_id, self.rule_name, True, 20, "match", (evidence,))


class FailureRule(BaseRule):
    rule_id = "failure"
    rule_name = "Failure Rule"
    description = "Failure risk"

    def __init__(self):
        self.calls = 0

    def evaluate(self, transactions):
        self.calls += 1
        raise RuntimeError("controlled failure")


@pytest.fixture
def setup():
    connection = sqlite3.connect(":memory:")
    repository = FdsResultRepository(connection, alert_id_factory=lambda: "alert-fixed")
    repository.initialize_schema()
    service = FdsAnalysisService(
        repository, source_name="sample.csv",
        ruleset_version="bank-fds-default-v1+69106de",
    )
    yield service, repository, connection
    connection.close()


def test_single_custom_rule_execution_and_typed_result(setup):
    service, _, _ = setup
    rule = CountingRule(triggered=True)
    result = service.analyze_and_persist(
        pd.DataFrame([_row()]), rules=[rule], analysis_run_id="run-1", created_at=CREATED
    )
    assert isinstance(result, AnalysisServiceResult)
    assert rule.calls == 1
    assert result.prediction["fraud_score"] == result.artifact.fraud_score == .2
    assert result.analysis_run.fraud_score == result.alert.summary.fraud_score == .2
    assert result.prediction["triggered_rules"] == ["counting"]
    assert result.alert.triggered_rule_ids == ("counting",)


def test_clean_rules_empty_and_repository_stays_open(setup):
    service, repository, connection = setup
    result = service.analyze_and_persist(
        pd.DataFrame([_row()]), rules=[], analysis_run_id="clean", created_at=CREATED
    )
    assert result.prediction["fraud_score"] == result.artifact.fraud_score == 0.0
    assert result.artifact.rule_findings == () and result.alert is None
    assert connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 1
    assert repository.get_analysis_run("clean") is not None


def test_rules_none_uses_default_five_and_custom_replaces(setup):
    service, _, _ = setup
    default = service.analyze_and_persist(
        pd.DataFrame([_row()]), analysis_run_id="default", created_at=CREATED
    )
    custom = service.analyze_and_persist(
        pd.DataFrame([_row()]), rules=[CountingRule()],
        analysis_run_id="custom", created_at=CREATED,
    )
    assert len(default.artifact.rule_findings) == 5
    assert [value.rule_id for value in custom.artifact.rule_findings] == ["counting"]


def test_input_dataframe_and_prediction_snapshot_are_isolated(setup):
    service, _, _ = setup
    frame = pd.DataFrame([_row(), _row(step=2)], index=["dup", "dup"])
    before = frame.copy(deep=True)
    result = service.analyze_and_persist(
        frame, rules=[], analysis_run_id="run", created_at=CREATED
    )
    assert_frame_equal(frame, before, check_dtype=True)
    result.prediction["fraud_score"] = 1.0
    result.prediction["details"]["injected"] = True
    assert result.artifact.fraud_score == result.analysis_run.fraud_score == 0.0


def test_repeated_calls_and_duplicate_run_id(setup):
    service, _, connection = setup
    frame = pd.DataFrame([_row()])
    service.analyze_and_persist(frame, rules=[], analysis_run_id="run-1", created_at=CREATED)
    service.analyze_and_persist(frame, rules=[], analysis_run_id="run-2", created_at=CREATED)
    with pytest.raises(sqlite3.IntegrityError):
        service.analyze_and_persist(frame, rules=[], analysis_run_id="run-1", created_at=CREATED)
    assert connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 2


def test_closed_repository_error_is_propagated(setup):
    service, repository, _ = setup
    repository.close()
    with pytest.raises(RepositoryClosedError):
        service.analyze_and_persist(
            pd.DataFrame([_row()]), rules=[], analysis_run_id="run", created_at=CREATED
        )


def test_persistence_failure_rolls_back_without_reexecution(setup):
    service, _, connection = setup
    connection.execute(
        """CREATE TRIGGER abort_findings BEFORE INSERT ON rule_findings
           BEGIN SELECT RAISE(ABORT, 'save failure'); END"""
    )
    rule = CountingRule(triggered=True)
    frame = pd.DataFrame([_row()]); before = frame.copy(deep=True)
    with pytest.raises(sqlite3.IntegrityError, match="save failure"):
        service.analyze_and_persist(
            frame, rules=[rule], analysis_run_id="failed", created_at=CREATED
        )
    assert rule.calls == 1
    assert connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 0
    assert_frame_equal(frame, before)


def test_rule_failure_is_saved_with_successful_finding_and_service_succeeds(setup):
    service, _, connection = setup
    success = CountingRule(triggered=True); failure = FailureRule()
    result = service.analyze_and_persist(
        pd.DataFrame([_row()]), rules=[success, failure],
        analysis_run_id="mixed", created_at=CREATED,
    )
    assert success.calls == failure.calls == 1
    assert result.prediction["details"]["skipped_rules"] == {
        "failure": "controlled failure"
    }
    assert result.artifact.rule_errors[0].rule_id == "failure"
    assert result.alert.errors[0].rule_id == "failure"
    assert connection.execute("SELECT count(*) FROM rule_execution_errors").fetchone()[0] == 1


def test_read_back_integrity_failure_is_distinct_and_saved_rows_remain(setup, monkeypatch):
    service, repository, connection = setup
    monkeypatch.setattr(repository, "get_analysis_run", lambda run_id: None)
    with pytest.raises(AnalysisServiceIntegrityError, match="run"):
        service.analyze_and_persist(
            pd.DataFrame([_row()]), rules=[], analysis_run_id="saved", created_at=CREATED
        )
    assert connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 1


def test_missing_expected_alert_is_read_back_integrity_failure(setup, monkeypatch):
    service, repository, connection = setup
    monkeypatch.setattr(repository, "get_alert_by_run_id", lambda run_id: None)
    with pytest.raises(AnalysisServiceIntegrityError, match="alert"):
        service.analyze_and_persist(
            pd.DataFrame([_row()]), rules=[CountingRule(triggered=True)],
            analysis_run_id="saved-risk", created_at=CREATED,
        )
    assert connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 1
    assert connection.execute("SELECT count(*) FROM alerts").fetchone()[0] == 1


def test_adapter_failure_prevents_save(setup):
    service, _, connection = setup
    with pytest.raises(ValueError):
        service.analyze_and_persist(
            pd.DataFrame({"wrong": [1]}), analysis_run_id="bad", created_at=CREATED
        )
    assert connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 0
