"""Immutable typed views of persisted bank FDS analysis results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AnalysisRunRecord:
    analysis_run_id: str
    source_name: str
    ruleset_version: str
    created_at: datetime
    input_row_count: int
    fraud_score: float
    transaction_count: int
    finding_count: int
    error_count: int


@dataclass(frozen=True, slots=True)
class AlertSummary:
    alert_id: str
    analysis_run_id: str
    fraud_score: float
    risk_level: str | None
    status: str
    created_at: datetime
    triggered_rule_count: int


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    canonical_transaction_id: str
    source_position: int
    source_row_id: str | int | None
    step: int | None
    transaction_datetime: datetime | None
    action: str
    amount: float
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
    evidence_order: int


@dataclass(frozen=True, slots=True)
class RuleFindingRecord:
    rule_id: str
    rule_name: str
    risk_type: str
    triggered: bool
    rule_score: int
    reason: str
    execution_order: int
    evidence: tuple[EvidenceRecord, ...]


@dataclass(frozen=True, slots=True)
class RuleExecutionErrorRecord:
    rule_id: str
    rule_name: str
    error_type: str
    message: str
    execution_order: int | None


@dataclass(frozen=True, slots=True)
class AlertDetail:
    summary: AlertSummary
    analysis_run: AnalysisRunRecord
    findings: tuple[RuleFindingRecord, ...]
    errors: tuple[RuleExecutionErrorRecord, ...]

    @property
    def triggered_rule_ids(self) -> tuple[str, ...]:
        return tuple(value.rule_id for value in self.findings if value.triggered)
