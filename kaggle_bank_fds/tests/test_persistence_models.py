"""Contracts for immutable bank FDS persistence snapshots."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import math

import pytest

from kaggle_bank_fds.src.persistence.persistence_models import (
    AnalysisPersistenceArtifact,
    RuleExecutionErrorSnapshot,
    RuleFindingSnapshot,
    TransactionSnapshot,
)


def _transaction(transaction_id="tx-1", source_position=0, source_row_id=0, **changes):
    values = dict(
        canonical_transaction_id=transaction_id,
        source_row_id=source_row_id,
        source_position=source_position,
        step=1,
        transaction_datetime=None,
        action_type="TRANSFER",
        amount=100.0,
        actor_account="A",
        target_account="B",
        counterparty_name=None,
        balance_before=200.0,
        balance_after=100.0,
        target_balance_before=0.0,
        target_balance_after=100.0,
        description=None,
        bank_name="PaySim",
        source_format="PAYSIM",
    )
    values.update(changes)
    return TransactionSnapshot(**values)


def _finding(rule_id="rule", order=0, triggered=True, evidence=("tx-1",)):
    return RuleFindingSnapshot(
        rule_id=rule_id,
        rule_name="Rule",
        risk_type="Risk",
        triggered=triggered,
        rule_score=20 if triggered else 0,
        reason="Reason",
        execution_order=order,
        evidence_transaction_ids=evidence,
    )


def _artifact(**changes):
    values = dict(
        analysis_run_id="run-1",
        source_name="sample.csv",
        ruleset_version="rules-v1",
        created_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
        input_row_count=1,
        fraud_score=0.2,
        transactions=(_transaction(),),
        rule_findings=(_finding(),),
        rule_errors=(),
    )
    values.update(changes)
    return AnalysisPersistenceArtifact(**values)


def test_models_are_frozen_and_mutable_inputs_are_snapshotted():
    transactions = [_transaction()]
    findings = [_finding()]
    artifact = _artifact(transactions=transactions, rule_findings=findings)
    transactions.clear(); findings.clear()
    assert len(artifact.transactions) == len(artifact.rule_findings) == 1
    with pytest.raises(FrozenInstanceError):
        artifact.fraud_score = 0.5


@pytest.mark.parametrize("field", ["analysis_run_id", "source_name", "ruleset_version"])
def test_artifact_rejects_blank_required_strings(field):
    with pytest.raises(ValueError):
        _artifact(**{field: " "})


@pytest.mark.parametrize("score", [True, math.nan, math.inf, -math.inf, -0.1, 1.1])
def test_artifact_rejects_invalid_fraud_score(score):
    with pytest.raises((TypeError, ValueError)):
        _artifact(fraud_score=score)


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_artifact_accepts_fraud_score_boundaries(score):
    assert _artifact(fraud_score=score).fraud_score == score


def test_artifact_requires_timezone_aware_created_at_and_normalizes_utc():
    with pytest.raises(ValueError, match="timezone"):
        _artifact(created_at=datetime(2026, 8, 4))
    assert _artifact().created_at.tzinfo is timezone.utc


def test_duplicate_transaction_rule_and_execution_order_are_rejected():
    duplicate = _transaction("tx-1", 1, 1)
    with pytest.raises(ValueError, match="transaction IDs"):
        _artifact(input_row_count=2, transactions=(_transaction(), duplicate))
    with pytest.raises(ValueError, match="finding rule IDs"):
        _artifact(rule_findings=(_finding(), _finding()))
    error = RuleExecutionErrorSnapshot("error", "Error", "RuntimeError", "safe", 0)
    with pytest.raises(ValueError, match="execution_order"):
        _artifact(rule_errors=(error,))


def test_finding_and_error_for_same_rule_are_rejected():
    error = RuleExecutionErrorSnapshot("rule", "Rule", "RuntimeError", "safe", 1)
    with pytest.raises(ValueError, match="both successful and failed"):
        _artifact(rule_errors=(error,))


def test_invalid_and_duplicate_evidence_rejected():
    with pytest.raises(ValueError, match="unique"):
        _finding(evidence=("tx-1", "tx-1"))
    with pytest.raises(ValueError, match="reference"):
        _artifact(rule_findings=(_finding(evidence=("unknown",)),))


def test_triggered_and_clean_finding_invariants():
    with pytest.raises(ValueError, match="score and evidence"):
        _finding(evidence=())
    with pytest.raises(ValueError, match="zero score"):
        RuleFindingSnapshot("r", "R", "Risk", False, 1, "reason", 0, ())
    assert _finding(triggered=False, evidence=()).rule_score == 0


@pytest.mark.parametrize("score", [True, -1, 101, 1.5])
def test_rule_score_validation(score):
    with pytest.raises((TypeError, ValueError)):
        RuleFindingSnapshot("r", "R", "Risk", True, score, "reason", 0, ("tx-1",))


def test_source_row_id_and_numeric_validation():
    assert _transaction(source_row_id=" row ").source_row_id == "row"
    with pytest.raises(TypeError, match="source_row_id"):
        _transaction(source_row_id=True)
    with pytest.raises((TypeError, ValueError)):
        _transaction(amount=math.inf)


def test_error_message_is_redacted_to_single_line_and_bounded():
    error = RuleExecutionErrorSnapshot(
        "rule", "Rule", "RuntimeError", "line1\nline2\x00" + "x" * 600, 0
    )
    assert "\n" not in error.message and "\x00" not in error.message
    assert len(error.message) == 500
