"""Integration contracts for lossless Plugin analysis artifacts."""

from datetime import datetime, timezone
import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.adapters.paysim_adapter import PaySimAdapter
from kaggle_bank_fds.src.models.plugin_rule_predictor import predict_with_plugins
from kaggle_bank_fds.src.persistence.analysis_artifact_builder import (
    analyze_with_plugins_for_persistence,
    build_analysis_artifact,
)
from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.evidence_item import EvidenceItem
from kaggle_bank_fds.src.rules.rule_engine_report import RuleEngineReport
from kaggle_bank_fds.src.rules.rule_execution_error import RuleExecutionError
from kaggle_bank_fds.src.rules.rule_result import RuleResult


CREATED_AT = datetime(2026, 8, 4, tzinfo=timezone.utc)


def _row(step=1, amount=100.0, target="M", action="PAYMENT", actor="A", balance=500_000.0):
    return {
        "step": step, "type": action, "amount": amount, "nameOrig": actor,
        "oldbalanceOrg": balance, "newbalanceOrig": balance - amount,
        "nameDest": target, "oldbalanceDest": 0.0, "newbalanceDest": amount,
    }


def _analyze(frame, **changes):
    options = dict(
        source_name="sample.csv",
        ruleset_version="bank-fds-default-v1+69106de",
        analysis_run_id="run-fixed",
        created_at=CREATED_AT,
    )
    options.update(changes)
    return analyze_with_plugins_for_persistence(frame, **options)


def test_clean_default_analysis_has_five_clean_findings_and_same_facade():
    frame = pd.DataFrame([_row()]); before = frame.copy(deep=True)
    prediction, artifact = _analyze(frame)
    assert prediction == predict_with_plugins(frame)
    assert list(prediction) == ["fraud_score", "triggered_rules", "details"]
    assert artifact.fraud_score == 0.0
    assert len(artifact.transactions) == 1
    assert [finding.rule_id for finding in artifact.rule_findings] == [
        "transfer_cash_out", "full_balance_transfer", "rounded_amount",
        "rapid_repeated_transfer", "split_transaction",
    ]
    assert all(not finding.triggered for finding in artifact.rule_findings)
    assert_frame_equal(frame, before)


def test_exact_overlap_preserves_canonical_evidence_despite_duplicate_raw_index():
    frame = pd.DataFrame([
        _row(1, 70_000, "B", "TRANSFER"),
        _row(2, 70_000, "B", "TRANSFER"),
        _row(3, 70_000, "B", "TRANSFER"),
    ], index=["same", "same", "same"])
    prediction, artifact = _analyze(frame)
    findings = {finding.rule_id: finding for finding in artifact.rule_findings}
    rapid = findings["rapid_repeated_transfer"]
    split = findings["split_transaction"]
    assert prediction["fraud_score"] == artifact.fraud_score == 0.35
    assert rapid.rule_score == 30 and split.rule_score == 35
    assert rapid.evidence_transaction_ids == split.evidence_transaction_ids
    assert len(rapid.evidence_transaction_ids) == 3
    assert len(set(rapid.evidence_transaction_ids)) == 3
    assert prediction["details"]["rapid_repeated_transfer"]["evidence_ids"] == ["same"]


def test_partial_overlap_preserves_score_and_evidence_order():
    frame = pd.DataFrame([
        _row(1, 70_000, "B", "TRANSFER"),
        _row(2, 70_000, "B", "TRANSFER"),
        _row(3, 70_000, "B", "TRANSFER"),
        _row(4, 10_000, "C", "TRANSFER"),
    ])
    prediction, artifact = _analyze(frame)
    findings = {finding.rule_id: finding for finding in artifact.rule_findings}
    assert prediction["fraud_score"] == artifact.fraud_score == 0.65
    assert len(findings["rapid_repeated_transfer"].evidence_transaction_ids) == 3
    assert len(findings["split_transaction"].evidence_transaction_ids) == 4
    assert findings["split_transaction"].evidence_transaction_ids == tuple(
        transaction.canonical_transaction_id for transaction in artifact.transactions
    )


class FailureRule(BaseRule):
    rule_id = "failure"; rule_name = "Failure"; description = "Failure risk"
    def evaluate(self, transactions): raise RuntimeError("unsafe\nmessage")


def test_failed_rule_preserves_error_and_success_order():
    frame = pd.DataFrame([_row()])
    prediction, artifact = _analyze(frame, rules=[FailureRule()])
    assert prediction["details"]["skipped_rules"] == {"failure": "unsafe\nmessage"}
    assert artifact.rule_findings == ()
    assert len(artifact.rule_errors) == 1
    error = artifact.rule_errors[0]
    assert (error.rule_id, error.error_type, error.execution_order) == (
        "failure", "RuntimeError", 0,
    )
    assert error.message == "unsafe message"


def test_uuid_and_time_injection_are_reproducible():
    frame = pd.DataFrame([_row()])
    first = _analyze(frame)[1]; second = _analyze(frame)[1]
    assert first == second


def test_source_id_normalization_and_missing_values():
    canonical = PaySimAdapter().transform(pd.DataFrame([_row(), _row(step=2)]))
    canonical["source_row_id"] = pd.Series([np.int64(7), pd.NA], dtype="object")
    report = RuleEngineReport()
    artifact = build_analysis_artifact(
        canonical_transactions=canonical, engine_report=report, fraud_score=0.0,
        source_name="sample", ruleset_version="v1", rule_risk_types={},
        rule_execution_order={}, analysis_run_id="run", created_at=CREATED_AT,
    )
    assert [tx.source_row_id for tx in artifact.transactions] == [7, None]
    assert all(type(tx.source_row_id) in (int, type(None)) for tx in artifact.transactions)


def test_duplicate_canonical_id_and_missing_column_are_rejected():
    canonical = PaySimAdapter().transform(pd.DataFrame([_row(), _row(step=2)]))
    duplicate = canonical.copy(); duplicate.loc[:, "transaction_id"] = "duplicate"
    kwargs = dict(engine_report=RuleEngineReport(), fraud_score=0.0,
                  source_name="sample", ruleset_version="v1", rule_risk_types={},
                  rule_execution_order={}, analysis_run_id="run", created_at=CREATED_AT)
    with pytest.raises(ValueError, match="transaction IDs"):
        build_analysis_artifact(canonical_transactions=duplicate, **kwargs)
    with pytest.raises(ValueError, match="Missing canonical"):
        build_analysis_artifact(canonical_transactions=canonical.drop(columns="amount"), **kwargs)


def test_unknown_and_duplicate_evidence_ids_are_rejected():
    canonical = PaySimAdapter().transform(pd.DataFrame([_row()]))
    evidence = EvidenceItem("unknown", 0, None, None, None, 1.0, "evidence")
    result = RuleResult("rule", "Rule", True, 20, "reason", (evidence,))
    with pytest.raises(ValueError, match="reference"):
        build_analysis_artifact(
            canonical_transactions=canonical,
            engine_report=RuleEngineReport(results=(result,)), fraud_score=0.2,
            source_name="sample", ruleset_version="v1",
            rule_risk_types={"rule": "Risk"}, rule_execution_order={"rule": 0},
            analysis_run_id="run", created_at=CREATED_AT,
        )


@pytest.mark.parametrize("bad_score", [True, math.nan, math.inf, -0.1, 1.1])
def test_builder_rejects_invalid_score(bad_score):
    canonical = PaySimAdapter().transform(pd.DataFrame([_row()]))
    with pytest.raises((TypeError, ValueError)):
        build_analysis_artifact(
            canonical_transactions=canonical, engine_report=RuleEngineReport(),
            fraud_score=bad_score, source_name="sample", ruleset_version="v1",
            rule_risk_types={}, rule_execution_order={}, analysis_run_id="run",
            created_at=CREATED_AT,
        )


def test_builder_inputs_are_not_mutated():
    canonical = PaySimAdapter().transform(pd.DataFrame([_row()]))
    before = canonical.copy(deep=True); report = RuleEngineReport()
    build_analysis_artifact(
        canonical_transactions=canonical, engine_report=report, fraud_score=0.0,
        source_name="sample", ruleset_version="v1", rule_risk_types={},
        rule_execution_order={}, analysis_run_id="run", created_at=CREATED_AT,
    )
    assert_frame_equal(canonical, before); assert report == RuleEngineReport()
