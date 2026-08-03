"""Immutable, database-agnostic snapshots of one bank FDS analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from numbers import Integral, Real
import re


SourceRowId = str | int | None
_MAX_ERROR_MESSAGE_LENGTH = 500


@dataclass(frozen=True, slots=True)
class TransactionSnapshot:
    canonical_transaction_id: str
    source_row_id: SourceRowId
    source_position: int
    step: int | None
    transaction_datetime: datetime | None
    action_type: str
    amount: float
    actor_account: str
    target_account: str
    counterparty_name: str | None
    balance_before: float | None
    balance_after: float | None
    target_balance_before: float | None
    target_balance_after: float | None
    description: str | None
    bank_name: str
    source_format: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_transaction_id", _required_string(
            self.canonical_transaction_id, "canonical_transaction_id"
        ))
        object.__setattr__(self, "source_row_id", _source_row_id(self.source_row_id))
        object.__setattr__(self, "source_position", _nonnegative_int(
            self.source_position, "source_position"
        ))
        if self.step is not None:
            object.__setattr__(self, "step", _integer(self.step, "step"))
        if self.transaction_datetime is not None and not isinstance(
            self.transaction_datetime, datetime
        ):
            raise TypeError("transaction_datetime must be datetime or None.")
        for field_name in ("action_type", "actor_account", "target_account",
                           "bank_name", "source_format"):
            object.__setattr__(self, field_name, _required_string(
                getattr(self, field_name), field_name
            ))
        for field_name in ("counterparty_name", "description"):
            object.__setattr__(self, field_name, _optional_string(
                getattr(self, field_name), field_name
            ))
        object.__setattr__(self, "amount", _finite_float(self.amount, "amount"))
        for field_name in ("balance_before", "balance_after",
                           "target_balance_before", "target_balance_after"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _finite_float(value, field_name))


@dataclass(frozen=True, slots=True)
class RuleFindingSnapshot:
    rule_id: str
    rule_name: str
    risk_type: str
    triggered: bool
    rule_score: int
    reason: str
    execution_order: int
    evidence_transaction_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("rule_id", "rule_name", "risk_type", "reason"):
            object.__setattr__(self, field_name, _required_string(
                getattr(self, field_name), field_name
            ))
        if type(self.triggered) is not bool:
            raise TypeError("triggered must be bool.")
        score = _integer(self.rule_score, "rule_score")
        if not 0 <= score <= 100:
            raise ValueError("rule_score must be between 0 and 100.")
        object.__setattr__(self, "rule_score", score)
        object.__setattr__(self, "execution_order", _nonnegative_int(
            self.execution_order, "execution_order"
        ))
        evidence = tuple(
            _required_string(value, "evidence_transaction_id")
            for value in self.evidence_transaction_ids
        )
        if len(evidence) != len(set(evidence)):
            raise ValueError("evidence transaction IDs must be unique.")
        if self.triggered:
            if score < 1 or not evidence:
                raise ValueError("triggered findings require score and evidence.")
        elif score != 0 or evidence:
            raise ValueError("clean findings require zero score and no evidence.")
        object.__setattr__(self, "evidence_transaction_ids", evidence)


@dataclass(frozen=True, slots=True)
class RuleExecutionErrorSnapshot:
    rule_id: str
    rule_name: str
    error_type: str
    message: str
    execution_order: int

    def __post_init__(self) -> None:
        for field_name in ("rule_id", "rule_name", "error_type"):
            object.__setattr__(self, field_name, _required_string(
                getattr(self, field_name), field_name
            ))
        object.__setattr__(self, "message", _safe_error_message(self.message))
        object.__setattr__(self, "execution_order", _nonnegative_int(
            self.execution_order, "execution_order"
        ))


@dataclass(frozen=True, slots=True)
class AnalysisPersistenceArtifact:
    analysis_run_id: str
    source_name: str
    ruleset_version: str
    created_at: datetime
    input_row_count: int
    fraud_score: float
    transactions: tuple[TransactionSnapshot, ...]
    rule_findings: tuple[RuleFindingSnapshot, ...]
    rule_errors: tuple[RuleExecutionErrorSnapshot, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("analysis_run_id", "source_name", "ruleset_version"):
            object.__setattr__(self, field_name, _required_string(
                getattr(self, field_name), field_name
            ))
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be datetime.")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware.")
        object.__setattr__(self, "created_at", self.created_at.astimezone(timezone.utc))
        count = _nonnegative_int(self.input_row_count, "input_row_count")
        object.__setattr__(self, "input_row_count", count)
        score = _finite_float(self.fraud_score, "fraud_score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("fraud_score must be between 0.0 and 1.0.")
        object.__setattr__(self, "fraud_score", score)
        transactions = tuple(self.transactions)
        findings = tuple(self.rule_findings)
        errors = tuple(self.rule_errors)
        if not all(isinstance(value, TransactionSnapshot) for value in transactions):
            raise TypeError("transactions must contain TransactionSnapshot values.")
        if not all(isinstance(value, RuleFindingSnapshot) for value in findings):
            raise TypeError("rule_findings must contain RuleFindingSnapshot values.")
        if not all(isinstance(value, RuleExecutionErrorSnapshot) for value in errors):
            raise TypeError("rule_errors must contain RuleExecutionErrorSnapshot values.")
        if len(transactions) != count:
            raise ValueError("transaction count must equal input_row_count.")
        transaction_ids = [value.canonical_transaction_id for value in transactions]
        if len(transaction_ids) != len(set(transaction_ids)):
            raise ValueError("canonical transaction IDs must be unique.")
        finding_ids = [value.rule_id for value in findings]
        error_ids = [value.rule_id for value in errors]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("finding rule IDs must be unique.")
        if len(error_ids) != len(set(error_ids)):
            raise ValueError("error rule IDs must be unique.")
        if set(finding_ids) & set(error_ids):
            raise ValueError("a rule cannot be both successful and failed.")
        orders = [value.execution_order for value in (*findings, *errors)]
        if len(orders) != len(set(orders)):
            raise ValueError("rule execution_order values must be unique.")
        known_ids = set(transaction_ids)
        for finding in findings:
            if not set(finding.evidence_transaction_ids) <= known_ids:
                raise ValueError("finding evidence must reference artifact transactions.")
        object.__setattr__(self, "transactions", transactions)
        object.__setattr__(self, "rule_findings", findings)
        object.__setattr__(self, "rule_errors", errors)


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")
    return value.strip() or None


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a bool-free integer.")
    return int(value)


def _nonnegative_int(value: object, name: str) -> int:
    normalized = _integer(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return normalized


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a bool-free real number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _source_row_id(value: object) -> SourceRowId:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("source_row_id must not be bool.")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        return value.strip()
    raise TypeError("source_row_id must be str, int, or None.")


def _safe_error_message(value: object) -> str:
    message = _required_string(value, "message")
    message = re.sub(r"[\x00-\x1f\x7f]", " ", message)
    message = " ".join(message.split())
    if not message:
        raise ValueError("message must not be empty.")
    return message[:_MAX_ERROR_MESSAGE_LENGTH]
