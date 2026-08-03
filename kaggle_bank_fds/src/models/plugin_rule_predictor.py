"""Legacy-compatible facade for running Plugin Rules on raw PaySim data."""

from numbers import Integral

import numpy as np
import pandas as pd

from kaggle_bank_fds.src.adapters.paysim_adapter import PaySimAdapter
from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.full_balance_transfer_rule import (
    FullBalanceTransferRule,
)
from kaggle_bank_fds.src.rules.rule_engine import RuleEngine
from kaggle_bank_fds.src.rules.rule_registry import RuleRegistry
from kaggle_bank_fds.src.rules.rule_result import RuleResult as PluginRuleResult
from kaggle_bank_fds.src.rules.transfer_cash_out_rule import TransferCashOutRule
from shared.rules.base_fraud_rule import RuleResult as LegacyRuleResult
from shared.scoring.risk_scorer import RiskScorer


_COMPATIBILITY_RISK_TYPES = {
    "transfer_cash_out": "이체 후 현금화 의심",
    "full_balance_transfer": "계좌 전액 이체 의심",
}


def predict_with_plugins(
    transaction_data: pd.DataFrame,
    *,
    rules: list[BaseRule] | None = None,
) -> dict:
    """Run Plugin Rules on raw PaySim rows and return the Legacy dict shape."""

    if not isinstance(transaction_data, pd.DataFrame):
        raise TypeError("transaction_data must be a pandas DataFrame.")
    if transaction_data.empty:
        raise ValueError("transaction_data must not be empty.")

    active_rules = _default_rules() if rules is None else _validate_rules(rules)
    registry = RuleRegistry()
    for rule in active_rules:
        registry.register(rule)

    canonical = PaySimAdapter().transform(transaction_data)
    report = RuleEngine(registry).evaluate(canonical)

    legacy_results: list[LegacyRuleResult] = []
    details: dict[str, dict | dict[str, str]] = {}
    for result in report.results:
        rule = registry.get(result.rule_id)
        risk_type = _COMPATIBILITY_RISK_TYPES.get(
            result.rule_id,
            rule.description,
        )
        evidence_ids = _restore_evidence_ids(
            result,
            raw_index=transaction_data.index,
        )
        legacy_result = LegacyRuleResult(
            rule_name=result.rule_id,
            risk_type=risk_type,
            is_suspicious=result.triggered,
            risk_score=result.score,
            reason=result.reason,
            evidence_ids=evidence_ids,
        )
        legacy_results.append(legacy_result)
        details[result.rule_id] = {
            "risk_type": risk_type,
            "is_suspicious": result.triggered,
            "risk_score": result.score,
            "reason": result.reason,
            "evidence_ids": evidence_ids,
        }

    details["skipped_rules"] = {
        error.rule_id: error.message for error in report.errors
    }
    investigation = RiskScorer().aggregate("batch", legacy_results)

    return {
        "fraud_score": investigation.total_score / 100.0,
        "triggered_rules": [result.rule_name for result in investigation.findings],
        "details": details,
    }


def _default_rules() -> list[BaseRule]:
    return [TransferCashOutRule(), FullBalanceTransferRule()]


def _validate_rules(rules: object) -> list[BaseRule]:
    if not isinstance(rules, list):
        raise TypeError("rules must be a list of Plugin BaseRule instances.")
    if not all(isinstance(rule, BaseRule) for rule in rules):
        raise TypeError("rules must contain only Plugin BaseRule instances.")
    return list(rules)


def _restore_evidence_ids(
    result: PluginRuleResult,
    *,
    raw_index: pd.Index,
) -> list[object]:
    evidence_ids: list[object] = []
    for item in result.evidence:
        source_row_id = item.source_row_id
        if source_row_id is None:
            continue
        if isinstance(source_row_id, (bool, np.bool_)) or not isinstance(
            source_row_id,
            Integral,
        ):
            raise TypeError(
                "Plugin evidence source_row_id must be an integer or None."
            )
        position = int(source_row_id)
        if not 0 <= position < len(raw_index):
            raise ValueError(
                "Plugin evidence source_row_id is outside the raw input range."
            )
        raw_id = raw_index[position]
        if not _contains_equal_value(evidence_ids, raw_id):
            evidence_ids.append(raw_id)
    return evidence_ids


def _contains_equal_value(values: list[object], candidate: object) -> bool:
    for existing in values:
        if existing is candidate:
            return True
        try:
            comparison = existing == candidate
        except Exception:
            continue
        try:
            if bool(comparison):
                return True
        except (TypeError, ValueError):
            try:
                if bool(comparison.all()):
                    return True
            except (AttributeError, TypeError, ValueError):
                continue
    return False
