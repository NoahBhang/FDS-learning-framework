"""Pure typed-model presenters for the Bank FDS operations UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Sequence

from kaggle_bank_fds.src.persistence.persistence_read_models import (
    AlertDetail,
    AlertSummary,
    EvidenceRecord,
    RuleExecutionErrorRecord,
    RuleFindingRecord,
)
from kaggle_bank_fds.src.persistence.persistence_models import (
    RuleExecutionErrorSnapshot,
    RuleFindingSnapshot,
)
from kaggle_bank_fds.src.services.fds_analysis_service import AnalysisServiceResult


RISK_SCORE_CAPTION = (
    "규칙 기반 위험점수이며 사기 발생 확률이 아닙니다. "
    "여러 Rule 점수의 합성과 조정 결과이며 위험등급 보정은 적용하지 않습니다."
)
SYNTHETIC_EVIDENCE_CAPTION = "PaySim 합성 거래 데이터이며 금액과 계좌 ID는 실제 고객 정보가 아닙니다."
_KST = timezone(timedelta(hours=9))
_STATUS_TEXT = {
    "OPEN": "검토 대기",
    "REVIEWING": "검토 중",
    "CONFIRMED": "이상 확정",
    "DISMISSED": "정상 처리",
}
_RISK_TYPE_TEXT = {
    "transfer_cash_out": "이체 후 현금화 의심",
    "full_balance_transfer": "계좌 잔액 대부분 이체",
    "rounded_amount": "고액 라운드 금액 거래",
    "rapid_repeated_transfer": "단기간 반복 이체",
    "split_transaction": "분할 이체 의심",
}
_USER_REASON_TEXT = {
    "transfer_cash_out": (
        "이체 후 유사한 금액이 현금화된 거래 관계를 탐지했습니다.",
        "설정된 시간 범위 안에서 이체 후 현금화 패턴을 탐지하지 않았습니다.",
    ),
    "full_balance_transfer": (
        "이체 전 잔액의 대부분을 보낸 거래 {evidence_count}건을 탐지했습니다.",
        "이체 전 잔액의 대부분을 보내는 거래를 탐지하지 않았습니다.",
    ),
    "rounded_amount": (
        "설정된 기준 이상의 라운드 금액 거래 {evidence_count}건을 탐지했습니다.",
        "설정된 기준 이상의 라운드 금액 거래를 탐지하지 않았습니다.",
    ),
    "rapid_repeated_transfer": (
        "동일 계좌 사이에서 단기간 반복된 이체 거래 {evidence_count}건을 탐지했습니다.",
        "동일 계좌 사이의 단기간 반복 이체 패턴을 탐지하지 않았습니다.",
    ),
    "split_transaction": (
        "한 송신자가 여러 건으로 나누어 보낸 이체 거래 {evidence_count}건을 탐지했습니다.",
        "설정된 기준에 해당하는 분할 이체 패턴을 탐지하지 않았습니다.",
    ),
}


@dataclass(frozen=True, slots=True)
class AnalysisSummaryView:
    analysis_run_id: str
    alert_id: str | None
    score_text: str
    raw_score: float
    triggered_rule_count: int
    transaction_count: int
    finding_count: int
    error_count: int
    created_at_text: str
    source_name: str
    ruleset_version: str
    clean: bool
    status_text: str


@dataclass(frozen=True, slots=True)
class AlertSummaryView:
    alert_id: str
    analysis_run_id: str
    created_at_text: str
    risk_score_text: str
    raw_score: float
    status_code: str
    status_text: str
    risk_level_text: str
    triggered_rule_count: int


@dataclass(frozen=True, slots=True)
class FindingView:
    execution_order: int
    rule_id: str
    rule_name: str
    risk_type: str
    triggered: bool
    triggered_text: str
    rule_score_text: str
    rule_score_raw: int
    user_reason: str
    technical_reason: str
    evidence_count: int


@dataclass(frozen=True, slots=True)
class EvidenceRowView:
    evidence_order: int
    canonical_transaction_id: str
    source_row_id: str | int | None
    step: int | None
    transaction_datetime_text: str | None
    action: str
    amount_raw: float
    amount_text: str
    actor_account: str
    target_account: str | None
    counterparty_account: str | None
    old_balance_actor: float | None
    new_balance_actor: float | None
    old_balance_target: float | None
    new_balance_target: float | None
    description: str | None
    bank_code: str | None
    source_format: str | None


@dataclass(frozen=True, slots=True)
class EvidenceTableView:
    rows: tuple[EvidenceRowView, ...]
    total_count: int
    displayed_count: int
    truncated: bool
    caption: str


@dataclass(frozen=True, slots=True)
class ErrorView:
    execution_order: int | None
    execution_order_text: str
    rule_id: str
    rule_name: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class AlertDetailView:
    summary: AlertSummaryView
    analysis_run_id: str
    source_name: str
    ruleset_version: str
    transaction_count: int
    triggered_findings: tuple[FindingView, ...]
    clean_findings: tuple[FindingView, ...]
    errors: tuple[ErrorView, ...]
    triggered_rule_ids: tuple[str, ...]
    total_evidence_count: int


def format_risk_score(score: float) -> str:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score must be a bool-free real number.")
    normalized = float(score)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("score must be finite and between 0.0 and 1.0.")
    points = (Decimal(str(normalized)) * Decimal(100)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return f"{points}/100"


def to_analysis_summary(result: AnalysisServiceResult) -> AnalysisSummaryView:
    scores = [result.artifact.fraud_score, result.analysis_run.fraud_score]
    if result.alert is not None:
        scores.append(result.alert.summary.fraud_score)
    if any(value != result.prediction.get("fraud_score") for value in scores):
        raise ValueError("Analysis result scores are inconsistent.")
    created = _format_kst(result.analysis_run.created_at)
    alert_id = None if result.alert is None else result.alert.summary.alert_id
    clean = result.alert is None
    return AnalysisSummaryView(
        analysis_run_id=result.analysis_run.analysis_run_id,
        alert_id=alert_id,
        score_text=format_risk_score(result.analysis_run.fraud_score),
        raw_score=result.analysis_run.fraud_score,
        triggered_rule_count=len(result.prediction.get("triggered_rules", ())),
        transaction_count=result.analysis_run.transaction_count,
        finding_count=result.analysis_run.finding_count,
        error_count=result.analysis_run.error_count,
        created_at_text=created,
        source_name=result.analysis_run.source_name,
        ruleset_version=result.analysis_run.ruleset_version,
        clean=clean,
        status_text=("정상 분석 완료 · Alert 미생성" if clean else "위험 패턴 탐지 · Alert 생성"),
    )


def to_alert_summary(alert: AlertSummary) -> AlertSummaryView:
    try:
        status_text = _STATUS_TEXT[alert.status]
    except KeyError as exc:
        raise ValueError(f"Unknown alert status: {alert.status}") from exc
    return AlertSummaryView(
        alert_id=alert.alert_id,
        analysis_run_id=alert.analysis_run_id,
        created_at_text=_format_kst(alert.created_at),
        risk_score_text=format_risk_score(alert.fraud_score),
        raw_score=alert.fraud_score,
        status_code=alert.status,
        status_text=status_text,
        risk_level_text="미분류" if alert.risk_level is None else alert.risk_level,
        triggered_rule_count=alert.triggered_rule_count,
    )


def to_finding_views(
    findings: Sequence[RuleFindingRecord | RuleFindingSnapshot],
) -> tuple[FindingView, ...]:
    views = []
    for value in findings:
        evidence_count = len(
            value.evidence if isinstance(value, RuleFindingRecord)
            else value.evidence_transaction_ids
        )
        reason_pair = _USER_REASON_TEXT.get(value.rule_id)
        user_reason = (
            value.reason
            if reason_pair is None
            else reason_pair[0 if value.triggered else 1].format(
                evidence_count=evidence_count
            )
        )
        views.append(FindingView(
            execution_order=value.execution_order,
            rule_id=value.rule_id,
            rule_name=value.rule_name,
            risk_type=_RISK_TYPE_TEXT.get(value.rule_id, value.risk_type),
            triggered=value.triggered,
            triggered_text="탐지" if value.triggered else "정상",
            rule_score_text=f"{value.rule_score}/100",
            rule_score_raw=value.rule_score,
            user_reason=user_reason,
            technical_reason=value.reason,
            evidence_count=evidence_count,
        ))
    return tuple(views)


def to_evidence_table(
    evidence: Sequence[EvidenceRecord], *, limit: int = 200
) -> EvidenceTableView:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be a bool-free integer.")
    if limit <= 0:
        raise ValueError("limit must be positive.")
    total = len(evidence)
    displayed = min(total, limit)
    rows = tuple(_evidence_row(evidence[index]) for index in range(displayed))
    truncated = displayed < total
    count_caption = (
        f"전체 {total:,}건 중 처음 {displayed:,}건을 표시합니다."
        if truncated else f"전체 Evidence {total:,}건을 표시합니다."
    )
    return EvidenceTableView(
        rows=rows,
        total_count=total,
        displayed_count=displayed,
        truncated=truncated,
        caption=f"{count_caption} {SYNTHETIC_EVIDENCE_CAPTION}",
    )


def to_error_views(
    errors: Sequence[RuleExecutionErrorRecord | RuleExecutionErrorSnapshot],
) -> tuple[ErrorView, ...]:
    return tuple(
        ErrorView(
            execution_order=value.execution_order,
            execution_order_text=(
                "순서 미확인" if value.execution_order is None else str(value.execution_order)
            ),
            rule_id=value.rule_id,
            rule_name=value.rule_name,
            error_type=value.error_type,
            message=value.message,
        )
        for value in errors
    )


def to_alert_detail(detail: AlertDetail) -> AlertDetailView:
    finding_views = to_finding_views(detail.findings)
    triggered = tuple(value for value in finding_views if value.triggered)
    clean = tuple(value for value in finding_views if not value.triggered)
    return AlertDetailView(
        summary=to_alert_summary(detail.summary),
        analysis_run_id=detail.analysis_run.analysis_run_id,
        source_name=detail.analysis_run.source_name,
        ruleset_version=detail.analysis_run.ruleset_version,
        transaction_count=detail.analysis_run.transaction_count,
        triggered_findings=triggered,
        clean_findings=clean,
        errors=to_error_views(detail.errors),
        triggered_rule_ids=detail.triggered_rule_ids,
        total_evidence_count=sum(len(value.evidence) for value in detail.findings),
    )


def _evidence_row(value: EvidenceRecord) -> EvidenceRowView:
    return EvidenceRowView(
        evidence_order=value.evidence_order,
        canonical_transaction_id=value.canonical_transaction_id,
        source_row_id=value.source_row_id,
        step=value.step,
        transaction_datetime_text=(
            None if value.transaction_datetime is None else _format_kst(value.transaction_datetime)
        ),
        action=value.action,
        amount_raw=value.amount,
        amount_text=f"{value.amount:,.2f}",
        actor_account=value.actor_account,
        target_account=value.target_account,
        counterparty_account=value.counterparty_account,
        old_balance_actor=value.old_balance_actor,
        new_balance_actor=value.new_balance_actor,
        old_balance_target=value.old_balance_target,
        new_balance_target=value.new_balance_target,
        description=value.description,
        bank_code=value.bank_code,
        source_format=value.source_format,
    )


def _format_kst(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("datetime value is required.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime value must be timezone-aware.")
    return value.astimezone(_KST).strftime("%Y-%m-%d %H:%M:%S KST")
