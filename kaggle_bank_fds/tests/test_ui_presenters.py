"""Contracts for pure Bank FDS display presenters."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import math

import pytest

from kaggle_bank_fds.src.persistence.persistence_models import (
    AnalysisPersistenceArtifact,
    RuleFindingSnapshot,
    TransactionSnapshot,
)
from kaggle_bank_fds.src.persistence.persistence_read_models import (
    AlertDetail,
    AlertSummary,
    AnalysisRunRecord,
    EvidenceRecord,
    RuleExecutionErrorRecord,
    RuleFindingRecord,
)
from kaggle_bank_fds.src.services.fds_analysis_service import AnalysisServiceResult
from kaggle_bank_fds.src.ui.presenters import (
    RISK_SCORE_CAPTION,
    AlertDetailView,
    format_risk_score,
    to_alert_detail,
    to_alert_summary,
    to_analysis_summary,
    to_error_views,
    to_evidence_table,
    to_finding_views,
)


UTC = timezone.utc
CREATED = datetime(2026, 8, 4, tzinfo=UTC)


def _evidence(identifier="tx-1", order=0, source_id=10, transaction_datetime=CREATED):
    return EvidenceRecord(
        identifier, order, source_id, order + 1, transaction_datetime, "TRANSFER", 1234.5,
        "A", "B", "C", 2000.0, 765.5, 0.0, 1234.5, "desc", "PaySim", "PAYSIM", order,
    )


def _finding(rule_id="rule", order=0, triggered=True, score=20, evidence=None):
    if evidence is None:
        evidence = (_evidence(),) if triggered else ()
    return RuleFindingRecord(
        rule_id, rule_id.title(), "risk", triggered, score, "reason", order, tuple(evidence)
    )


def _run(score=.2):
    return AnalysisRunRecord("run", "source", "v1", CREATED, 1, score, 1, 1, 0)


def _alert(score=.2, status="OPEN", risk_level=None):
    return AlertSummary("alert", "run", score, risk_level, status, CREATED, 1)


def _artifact(score=.2):
    transaction = TransactionSnapshot(
        "tx-1", 0, 0, 1, None, "TRANSFER", 1.0, "A", "B", None,
        None, None, None, None, None, "PaySim", "PAYSIM",
    )
    finding = RuleFindingSnapshot("rule", "Rule", "risk", True, 20, "reason", 0, ("tx-1",))
    return AnalysisPersistenceArtifact(
        "run", "source", "v1", CREATED, 1, score, (transaction,), (finding,), ()
    )


def _result(score=.2, alert=True):
    detail = None
    if alert:
        detail = AlertDetail(_alert(score), _run(score), (_finding(),), ())
    return AnalysisServiceResult(
        {"fraud_score": score, "triggered_rules": ["rule"] if alert else [], "details": {}},
        _artifact(score) if alert else AnalysisPersistenceArtifact(
            "run", "source", "v1", CREATED, 1, 0.0, _artifact().transactions, (), ()
        ),
        _run(score), detail,
    )


@pytest.mark.parametrize("score,expected", [
    (0.0, "0/100"), (.01, "1/100"), (.35, "35/100"), (.65, "65/100"),
    (1.0, "100/100"), (.005, "1/100"),
])
def test_score_formatting_uses_half_up_points(score, expected):
    assert format_risk_score(score) == expected
    assert "%" not in expected


@pytest.mark.parametrize("score", [True, math.nan, math.inf, -.1, 1.1, "0.5"])
def test_score_formatting_rejects_invalid_values(score):
    with pytest.raises((TypeError, ValueError)):
        format_risk_score(score)


def test_score_caption_is_not_probability_language():
    assert "확률이 아닙니다" in RISK_SCORE_CAPTION and "%" not in RISK_SCORE_CAPTION


def test_analysis_summary_clean_risk_kst_and_immutability():
    risky = to_analysis_summary(_result())
    assert risky.alert_id == "alert" and not risky.clean
    assert risky.status_text == "위험 패턴 탐지 · Alert 생성"
    assert risky.created_at_text == "2026-08-04 09:00:00 KST"
    clean = to_analysis_summary(_result(0.0, alert=False))
    assert clean.clean and clean.alert_id is None and clean.score_text == "0/100"
    with pytest.raises(FrozenInstanceError):
        clean.clean = False


def test_analysis_summary_rejects_score_mismatch_and_naive_datetime():
    result = _result(); object.__setattr__(result.analysis_run, "fraud_score", .3)
    with pytest.raises(ValueError, match="inconsistent"):
        to_analysis_summary(result)
    run = _run(); object.__setattr__(run, "created_at", datetime(2026, 8, 4))
    result = _result(); object.__setattr__(result, "analysis_run", run)
    with pytest.raises(ValueError, match="timezone"):
        to_analysis_summary(result)


@pytest.mark.parametrize("code,text", [
    ("OPEN", "검토 대기"), ("REVIEWING", "검토 중"),
    ("CONFIRMED", "이상 확정"), ("DISMISSED", "정상 처리"),
])
def test_alert_status_mapping_and_nullable_risk(code, text):
    view = to_alert_summary(_alert(status=code))
    assert view.status_code == code and view.status_text == text
    assert view.risk_level_text == "미분류"


def test_unknown_alert_status_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        to_alert_summary(_alert(status="NEW"))


def test_finding_order_scores_reason_and_snapshot_support():
    findings = (_finding("rapid", 4, True, 30), _finding("split", 1, True, 35))
    views = to_finding_views(findings)
    assert [(v.rule_id, v.execution_order, v.rule_score_raw) for v in views] == [
        ("rapid", 4, 30), ("split", 1, 35),
    ]
    snapshot = RuleFindingSnapshot("clean", "Clean", "risk", False, 0, "safe", 2, ())
    assert to_finding_views((snapshot,))[0].evidence_count == 0


@pytest.mark.parametrize("rule_id,expected", [
    ("transfer_cash_out", "이체 후 현금화 의심"),
    ("full_balance_transfer", "계좌 잔액 대부분 이체"),
    ("rounded_amount", "고액 라운드 금액 거래"),
    ("rapid_repeated_transfer", "단기간 반복 이체"),
    ("split_transaction", "분할 이체 의심"),
])
def test_known_rule_risk_types_use_korean_presentation_mapping(rule_id, expected):
    view = to_finding_views((_finding(rule_id),))[0]
    assert view.risk_type == expected
    assert view.technical_reason == "reason"


def test_unknown_rule_preserves_risk_type_and_reason_without_information_loss():
    view = to_finding_views((_finding("custom_rule"),))[0]
    assert view.risk_type == "risk"
    assert view.user_reason == view.technical_reason == "reason"


def test_known_rule_user_reasons_distinguish_triggered_and_clean_contracts():
    triggered = to_finding_views((
        _finding("rounded_amount", evidence=(_evidence("a"), _evidence("b"))),
    ))[0]
    clean = to_finding_views((_finding("rounded_amount", triggered=False),))[0]
    assert triggered.user_reason == "설정된 기준 이상의 라운드 금액 거래 2건을 탐지했습니다."
    assert clean.user_reason == "설정된 기준 이상의 라운드 금액 거래를 탐지하지 않았습니다."
    assert triggered.technical_reason == clean.technical_reason == "reason"


def test_evidence_default_and_detailed_mapping_order_and_source_types():
    evidence = (
        _evidence("a", 2, 10), _evidence("a", 0, "10"),
        _evidence("b", 1, None, None),
    )
    view = to_evidence_table(evidence, limit=3)
    assert [row.canonical_transaction_id for row in view.rows] == ["a", "a", "b"]
    assert [row.source_row_id for row in view.rows] == [10, "10", None]
    assert view.rows[0].amount_raw == 1234.5 and view.rows[0].amount_text == "1,234.50"
    assert view.rows[2].transaction_datetime_text is None
    assert "합성" in view.caption


def test_evidence_limit_truncation_and_only_prefix_materialized():
    evidence = tuple(_evidence(f"tx-{index}", index) for index in range(10_000))
    view = to_evidence_table(evidence, limit=200)
    assert (view.total_count, view.displayed_count, view.truncated) == (10_000, 200, True)
    assert len(view.rows) == 200 and view.rows[-1].canonical_transaction_id == "tx-199"
    assert "10,000" in view.caption and "200" in view.caption


class _ObservedEvidence:
    def __init__(self, size):
        self.size = size
        self.requested = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.requested.append(index)
        return _evidence(f"tx-{index}", index)


@pytest.mark.parametrize("limit", [20, 200])
def test_large_evidence_materializes_only_selected_prefix(limit):
    evidence = _ObservedEvidence(10_000)
    view = to_evidence_table(evidence, limit=limit)
    assert view.total_count == 10_000
    assert view.displayed_count == len(view.rows) == limit
    assert evidence.requested == list(range(limit))
    assert [row.canonical_transaction_id for row in view.rows] == [
        f"tx-{index}" for index in range(limit)
    ]


@pytest.mark.parametrize("limit", [True, 1.5, "1", 0, -1])
def test_evidence_limit_validation(limit):
    error = TypeError if limit in (True, 1.5, "1") else ValueError
    with pytest.raises(error):
        to_evidence_table((_evidence(),), limit=limit)


def test_error_and_alert_detail_presenters_preserve_groups_and_duplicates():
    shared = _evidence("shared")
    findings = (
        _finding("rapid", 0, True, 30, (shared,)),
        _finding("clean", 1, False, 0, ()),
        _finding("split", 2, True, 35, (shared,)),
    )
    errors = (RuleExecutionErrorRecord("failed", "Failed", "Error", "safe", None),)
    detail = AlertDetail(_alert(.35), _run(.35), findings, errors)
    view = to_alert_detail(detail)
    assert isinstance(view, AlertDetailView)
    assert [item.rule_id for item in view.triggered_findings] == ["rapid", "split"]
    assert [item.rule_id for item in view.clean_findings] == ["clean"]
    assert view.total_evidence_count == 2
    assert view.triggered_rule_ids == ("rapid", "split")
    assert view.errors[0].execution_order_text == "순서 미확인"
