"""Build lossless persistence artifacts from Plugin Rule execution data."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from numbers import Integral, Real
from typing import Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from kaggle_bank_fds.src.adapters.canonical_transaction_schema import (
    CANONICAL_COLUMNS,
)
from kaggle_bank_fds.src.models.plugin_rule_predictor import (
    _execute_plugin_analysis,
)
from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.rule_engine_report import RuleEngineReport

from .persistence_models import (
    AnalysisPersistenceArtifact,
    RuleExecutionErrorSnapshot,
    RuleFindingSnapshot,
    TransactionSnapshot,
)


def build_analysis_artifact(
    *,
    canonical_transactions: pd.DataFrame,
    engine_report: RuleEngineReport,
    fraud_score: float,
    source_name: str,
    ruleset_version: str,
    rule_risk_types: Mapping[str, str],
    rule_execution_order: Mapping[str, int],
    analysis_run_id: str | None = None,
    created_at: datetime | None = None,
) -> AnalysisPersistenceArtifact:
    """Snapshot one completed analysis without performing persistence I/O."""

    if not isinstance(canonical_transactions, pd.DataFrame):
        raise TypeError("canonical_transactions must be a pandas DataFrame.")
    if not isinstance(engine_report, RuleEngineReport):
        raise TypeError("engine_report must be RuleEngineReport.")
    missing = [
        column for column in CANONICAL_COLUMNS
        if column not in canonical_transactions.columns
    ]
    if missing:
        raise ValueError("Missing canonical columns: " + ", ".join(missing))
    if not isinstance(rule_risk_types, Mapping):
        raise TypeError("rule_risk_types must be a mapping.")
    if not isinstance(rule_execution_order, Mapping):
        raise TypeError("rule_execution_order must be a mapping.")

    transactions = tuple(
        _transaction_snapshot(row, source_position)
        for source_position, (_, row) in enumerate(
            canonical_transactions.loc[:, CANONICAL_COLUMNS].iterrows()
        )
    )
    findings = tuple(
        RuleFindingSnapshot(
            rule_id=result.rule_id,
            rule_name=result.rule_name,
            risk_type=_mapping_string(rule_risk_types, result.rule_id, "risk type"),
            triggered=result.triggered,
            rule_score=result.score,
            reason=result.reason,
            execution_order=_mapping_order(rule_execution_order, result.rule_id),
            evidence_transaction_ids=tuple(
                item.transaction_id for item in result.evidence
            ),
        )
        for result in engine_report.results
    )
    errors = tuple(
        RuleExecutionErrorSnapshot(
            rule_id=error.rule_id,
            rule_name=error.rule_name,
            error_type=error.error_type,
            message=error.message,
            execution_order=_mapping_order(rule_execution_order, error.rule_id),
        )
        for error in engine_report.errors
    )
    return AnalysisPersistenceArtifact(
        analysis_run_id=str(uuid4()) if analysis_run_id is None else analysis_run_id,
        source_name=source_name,
        ruleset_version=ruleset_version,
        created_at=datetime.now(timezone.utc) if created_at is None else created_at,
        input_row_count=len(canonical_transactions),
        fraud_score=fraud_score,
        transactions=transactions,
        rule_findings=findings,
        rule_errors=errors,
    )


def analyze_with_plugins_for_persistence(
    transaction_data: pd.DataFrame,
    *,
    source_name: str,
    ruleset_version: str,
    rules: list[BaseRule] | None = None,
    analysis_run_id: str | None = None,
    created_at: datetime | None = None,
) -> tuple[dict, AnalysisPersistenceArtifact]:
    """Run the facade core once and return its dict plus a lossless artifact."""

    prediction, canonical, report, active_rules = _execute_plugin_analysis(
        transaction_data,
        rules=rules,
    )
    execution_order = {
        rule.rule_id: position for position, rule in enumerate(active_rules)
    }
    risk_types = {
        result.rule_id: prediction["details"][result.rule_id]["risk_type"]
        for result in report.results
    }
    artifact = build_analysis_artifact(
        canonical_transactions=canonical,
        engine_report=report,
        fraud_score=prediction["fraud_score"],
        source_name=source_name,
        ruleset_version=ruleset_version,
        rule_risk_types=risk_types,
        rule_execution_order=execution_order,
        analysis_run_id=analysis_run_id,
        created_at=created_at,
    )
    return prediction, artifact


def _transaction_snapshot(row: pd.Series, source_position: int) -> TransactionSnapshot:
    return TransactionSnapshot(
        canonical_transaction_id=_required_string(row["transaction_id"], "transaction_id"),
        source_row_id=_source_row_id(row["source_row_id"]),
        source_position=source_position,
        step=_optional_integer(row["step"], "step"),
        transaction_datetime=_optional_datetime(row["transaction_datetime"]),
        action_type=_required_string(row["action_type"], "action_type"),
        amount=_required_float(row["amount"], "amount"),
        actor_account=_required_string(row["actor_account"], "actor_account"),
        target_account=_required_string(row["target_account"], "target_account"),
        counterparty_name=_optional_string(row["counterparty_name"], "counterparty_name"),
        balance_before=_optional_float(row["balance_before"], "balance_before"),
        balance_after=_optional_float(row["balance_after"], "balance_after"),
        target_balance_before=_optional_float(
            row["target_balance_before"], "target_balance_before"
        ),
        target_balance_after=_optional_float(
            row["target_balance_after"], "target_balance_after"
        ),
        description=_optional_string(row["description"], "description"),
        bank_name=_required_string(row["bank_name"], "bank_name"),
        source_format=_required_string(row["source_format"], "source_format"),
    )


def _mapping_string(mapping: Mapping[str, str], key: str, label: str) -> str:
    try:
        return _required_string(mapping[key], label)
    except KeyError as error:
        raise ValueError(f"Missing {label} for rule_id: {key}") from error


def _mapping_order(mapping: Mapping[str, int], key: str) -> int:
    try:
        value = mapping[key]
    except KeyError as error:
        raise ValueError(f"Missing execution order for rule_id: {key}") from error
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("rule execution order must be a bool-free integer.")
    return int(value)


def _required_string(value: object, name: str) -> str:
    if _is_missing(value):
        raise ValueError(f"{name} must not be missing.")
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _optional_string(value: object, name: str) -> str | None:
    if _is_missing(value):
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or missing.")
    return value.strip() or None


def _source_row_id(value: object) -> str | int | None:
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("source_row_id must not be bool.")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        return value.strip()
    raise TypeError("source_row_id must be str, int, or missing.")


def _optional_integer(value: object, name: str) -> int | None:
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a bool-free integer or missing.")
    return int(value)


def _required_float(value: object, name: str) -> float:
    if _is_missing(value):
        raise ValueError(f"{name} must not be missing.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a bool-free real number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _optional_float(value: object, name: str) -> float | None:
    if _is_missing(value):
        return None
    return _required_float(value, name)


def _optional_datetime(value: object) -> datetime | None:
    if _is_missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if not isinstance(value, datetime):
        raise TypeError("transaction_datetime must be datetime or missing.")
    return value


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False
