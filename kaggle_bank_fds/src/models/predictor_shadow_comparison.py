"""Explicit, side-effect-free comparison of Legacy and Plugin predictions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
import math
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.models.plugin_rule_predictor import predict_with_plugins
from kaggle_bank_fds.src.models.predictor import predict


RuleState = Literal["success", "skipped", "missing"]
_DEFAULT_RULE_ORDER = ("transfer_cash_out", "full_balance_transfer")


class KnownDifference(str, Enum):
    ZERO_AMOUNT_TRANSFER_CASH_OUT = "zero_amount_transfer_cash_out"
    PLUGIN_STRICT_CANDIDATE_VALIDATION = "plugin_strict_candidate_validation"
    DUPLICATE_TRANSACTION_ID = "duplicate_transaction_id"
    NONFINITE_VALUE = "nonfinite_value"
    PAYSIM_ADAPTER_VALIDATION = "paysim_adapter_validation"
    MALFORMED_UNMATCHED_CANDIDATE = "malformed_unmatched_candidate"


@dataclass(frozen=True, slots=True)
class ExecutionFailure:
    stage: str
    error_type: str


@dataclass(frozen=True, slots=True)
class _MissingEvidenceValue:
    """Stable representation shared by scalar missing evidence values."""


@dataclass(frozen=True, slots=True)
class _ArrayEvidenceValue:
    dtype: str
    shape: tuple[int, ...]
    values: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class RuleComparison:
    rule_id: str
    legacy_state: RuleState
    plugin_state: RuleState
    legacy_triggered: bool | None
    plugin_triggered: bool | None
    legacy_score: int | float | None
    plugin_score: int | float | None
    triggered_matches: bool
    score_matches: bool
    risk_type_matches: bool
    reason_matches: bool
    evidence_matches: bool
    legacy_evidence_ids: tuple[object, ...]
    plugin_evidence_ids: tuple[object, ...]
    differences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ShadowComparisonReport:
    strict_equivalent: bool
    equivalent: bool
    total_score_matches: bool
    triggered_members_match: bool
    triggered_order_matches: bool
    skipped_rules_match: bool
    rule_comparisons: tuple[RuleComparison, ...]
    known_differences: tuple[KnownDifference, ...]
    unexpected_differences: tuple[str, ...]
    legacy_execution_failure: ExecutionFailure | None
    plugin_execution_failure: ExecutionFailure | None


def compare_legacy_and_plugin(
    transaction_data: pd.DataFrame,
) -> ShadowComparisonReport:
    """Run both predictors independently and return a redacted immutable report."""
    if not isinstance(transaction_data, pd.DataFrame):
        raise TypeError("transaction_data must be a pandas DataFrame.")
    if transaction_data.empty:
        raise ValueError("transaction_data must not be empty.")

    original_snapshot = transaction_data.copy(deep=True)
    legacy_input = transaction_data.copy(deep=True)
    plugin_input = transaction_data.copy(deep=True)
    legacy_snapshot = legacy_input.copy(deep=True)
    plugin_snapshot = plugin_input.copy(deep=True)

    legacy_result, legacy_failure = _execute(
        predict, legacy_input, "legacy_predictor"
    )
    assert_frame_equal(transaction_data, original_snapshot)
    assert_frame_equal(legacy_input, legacy_snapshot)

    plugin_result, plugin_failure = _execute(
        predict_with_plugins, plugin_input, "plugin_predictor"
    )
    assert_frame_equal(transaction_data, original_snapshot)
    assert_frame_equal(plugin_input, plugin_snapshot)

    if legacy_result is not None:
        _validate_result(legacy_result)
    if plugin_result is not None:
        _validate_result(plugin_result)

    comparisons = _compare_rules(legacy_result, plugin_result)
    complete = legacy_failure is None and plugin_failure is None
    total_matches = complete and legacy_result["fraud_score"] == plugin_result["fraud_score"]
    members_match = complete and Counter(legacy_result["triggered_rules"]) == Counter(
        plugin_result["triggered_rules"]
    )
    order_matches = complete and legacy_result["triggered_rules"] == plugin_result["triggered_rules"]
    skipped_match = complete and _skipped(legacy_result) == _skipped(plugin_result)

    mismatch_codes = _mismatch_codes(
        complete=complete,
        total_matches=total_matches,
        members_match=members_match,
        order_matches=order_matches,
        skipped_match=skipped_match,
        comparisons=comparisons,
    )
    known, explained = _classify_known(
        transaction_data,
        legacy_result,
        plugin_result,
        comparisons,
        mismatch_codes,
    )
    unexpected = tuple(code for code in mismatch_codes if code not in explained)

    strict = (
        complete
        and total_matches
        and members_match
        and order_matches
        and skipped_match
        and all(_strict_rule_match(item) for item in comparisons)
        and not unexpected
        and not known
    )
    equivalent = strict or (complete and bool(known) and not unexpected)
    return ShadowComparisonReport(
        strict_equivalent=strict,
        equivalent=equivalent,
        total_score_matches=total_matches,
        triggered_members_match=members_match,
        triggered_order_matches=order_matches,
        skipped_rules_match=skipped_match,
        rule_comparisons=comparisons,
        known_differences=known,
        unexpected_differences=unexpected,
        legacy_execution_failure=legacy_failure,
        plugin_execution_failure=plugin_failure,
    )


def _execute(function, frame: pd.DataFrame, stage: str):
    try:
        return function(frame), None
    except Exception as exc:
        return None, ExecutionFailure(stage=stage, error_type=type(exc).__name__)


def _validate_result(result: object) -> None:
    if not isinstance(result, dict) or not {"fraud_score", "triggered_rules", "details"} <= result.keys():
        raise TypeError("predictor result has an invalid top-level schema")
    score = result["fraud_score"]
    if isinstance(score, (bool, np.bool_)) or not isinstance(score, Real):
        raise TypeError("fraud_score must be a real number")
    if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
        raise ValueError("fraud_score must be finite and between zero and one")
    triggered = result["triggered_rules"]
    if not isinstance(triggered, list) or not all(isinstance(item, str) for item in triggered):
        raise TypeError("triggered_rules must be a list of strings")
    details = result["details"]
    if not isinstance(details, dict) or "skipped_rules" not in details:
        raise TypeError("details must contain skipped_rules")
    skipped = details["skipped_rules"]
    if not isinstance(skipped, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in skipped.items()
    ):
        raise TypeError("skipped_rules must map strings to strings")
    required = {"risk_type", "is_suspicious", "risk_score", "reason", "evidence_ids"}
    for rule_id, detail in details.items():
        if rule_id == "skipped_rules":
            continue
        if not isinstance(rule_id, str) or not isinstance(detail, dict) or not required <= detail.keys():
            raise TypeError("rule detail has an invalid schema")
        if type(detail["is_suspicious"]) is not bool:
            raise TypeError("is_suspicious must be bool")
        rule_score = detail["risk_score"]
        if isinstance(rule_score, (bool, np.bool_)) or not isinstance(rule_score, Real):
            raise TypeError("risk_score must be a real number")
        if not isinstance(detail["risk_type"], str) or not isinstance(detail["reason"], str):
            raise TypeError("risk_type and reason must be strings")
        if not isinstance(detail["evidence_ids"], list):
            raise TypeError("evidence_ids must be a list")


def _skipped(result: dict | None) -> dict[str, str]:
    return {} if result is None else result["details"]["skipped_rules"]


def _rule_ids(legacy: dict | None, plugin: dict | None) -> tuple[str, ...]:
    seen: list[str] = []
    for rule_id in _DEFAULT_RULE_ORDER:
        seen.append(rule_id)
    for result in (legacy, plugin):
        if result is None:
            continue
        for rule_id in (*result["details"].keys(), *_skipped(result).keys()):
            if rule_id != "skipped_rules" and rule_id not in seen:
                seen.append(rule_id)
    return tuple(seen)


def _state(result: dict | None, rule_id: str) -> RuleState:
    if result is None:
        return "missing"
    if rule_id in result["details"] and rule_id != "skipped_rules":
        return "success"
    if rule_id in _skipped(result):
        return "skipped"
    return "missing"


def _compare_rules(legacy: dict | None, plugin: dict | None) -> tuple[RuleComparison, ...]:
    output = []
    for rule_id in _rule_ids(legacy, plugin):
        legacy_state, plugin_state = _state(legacy, rule_id), _state(plugin, rule_id)
        left = legacy["details"][rule_id] if legacy_state == "success" else None
        right = plugin["details"][rule_id] if plugin_state == "success" else None
        left_evidence = _snapshot_evidence_ids(left["evidence_ids"]) if left else ()
        right_evidence = _snapshot_evidence_ids(right["evidence_ids"]) if right else ()
        triggered_matches = _optional_equal(left, right, "is_suspicious")
        score_matches = _optional_equal(left, right, "risk_score")
        risk_matches = _optional_equal(left, right, "risk_type")
        reason_matches = _optional_equal(left, right, "reason")
        evidence_matches = _sequence_equal(left_evidence, right_evidence)
        differences = []
        if legacy_state != plugin_state:
            differences.append("state_mismatch")
        for matches, code in (
            (triggered_matches, "triggered_mismatch"),
            (score_matches, "score_mismatch"),
            (risk_matches, "risk_type_mismatch"),
            (reason_matches, "reason_mismatch"),
            (evidence_matches, "evidence_mismatch"),
        ):
            if not matches:
                differences.append(code)
        if legacy_state == plugin_state == "skipped" and _skipped(legacy)[rule_id] != _skipped(plugin)[rule_id]:
            differences.append("skipped_message_mismatch")
        output.append(RuleComparison(
            rule_id=rule_id,
            legacy_state=legacy_state,
            plugin_state=plugin_state,
            legacy_triggered=left["is_suspicious"] if left else None,
            plugin_triggered=right["is_suspicious"] if right else None,
            legacy_score=left["risk_score"] if left else None,
            plugin_score=right["risk_score"] if right else None,
            triggered_matches=triggered_matches,
            score_matches=score_matches,
            risk_type_matches=risk_matches,
            reason_matches=reason_matches,
            evidence_matches=evidence_matches,
            legacy_evidence_ids=left_evidence,
            plugin_evidence_ids=right_evidence,
            differences=tuple(differences),
        ))
    return tuple(output)


def _optional_equal(left: dict | None, right: dict | None, key: str) -> bool:
    if left is None or right is None:
        return left is right
    return _safe_equal(left[key], right[key])


def _safe_equal(left: object, right: object) -> bool:
    if left is right:
        return True
    left_array = isinstance(left, np.ndarray)
    right_array = isinstance(right, np.ndarray)
    if left_array or right_array:
        if not left_array or not right_array or left.shape != right.shape:
            return False
        return all(
            _safe_equal(left_item, right_item)
            for left_item, right_item in zip(left.flat, right.flat)
        )
    left_missing = _is_missing_scalar(left)
    right_missing = _is_missing_scalar(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    try:
        comparison = left == right
    except Exception:
        return False
    if isinstance(comparison, (bool, np.bool_)):
        return bool(comparison)
    return False


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _snapshot_evidence_ids(values: list[object]) -> tuple[object, ...]:
    return tuple(_snapshot_evidence_value(value) for value in values)


def _snapshot_evidence_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _ArrayEvidenceValue(
            dtype=value.dtype.str,
            shape=tuple(value.shape),
            values=tuple(_snapshot_evidence_value(item) for item in value.flat),
        )
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot_evidence_value(item) for item in value)
    if _is_missing_scalar(value):
        return _MissingEvidenceValue()
    if isinstance(
        value,
        (str, bytes, int, float, complex, bool, type(None), date, datetime, Decimal),
    ):
        return value
    if isinstance(value, np.generic):
        return _snapshot_evidence_value(value.item())
    raise TypeError(
        f"unsupported mutable or non-scalar evidence ID type: {type(value).__name__}"
    )


def _sequence_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return len(left) == len(right) and all(_safe_equal(a, b) for a, b in zip(left, right))


def _strict_rule_match(item: RuleComparison) -> bool:
    return (
        item.legacy_state == item.plugin_state
        and item.triggered_matches
        and item.score_matches
        and item.risk_type_matches
        and item.evidence_matches
        and "skipped_message_mismatch" not in item.differences
    )


def _mismatch_codes(*, complete, total_matches, members_match, order_matches, skipped_match, comparisons):
    codes: list[str] = []
    if not complete:
        codes.append("execution_failure")
    for matches, code in (
        (total_matches, "total_score_mismatch"),
        (members_match, "triggered_members_mismatch"),
        (order_matches, "triggered_order_mismatch"),
        (skipped_match, "skipped_rules_mismatch"),
    ):
        if complete and not matches:
            codes.append(code)
    for item in comparisons:
        for difference in item.differences:
            if difference != "reason_mismatch":
                suffix = f"skipped_message_mismatch:{item.rule_id}" if difference == "skipped_message_mismatch" else f"{difference}:{item.rule_id}"
                if suffix not in codes:
                    codes.append(suffix)
    return tuple(codes)


def _classify_known(frame, legacy_result, plugin_result, comparisons, mismatch_codes):
    required = {"type", "amount", "nameOrig", "nameDest", "step"}
    if not required <= set(frame.columns) or not _has_zero_amount_pair(frame):
        return (), frozenset()
    transfer = next((item for item in comparisons if item.rule_id == "transfer_cash_out"), None)
    direct_codes = {
        "triggered_mismatch:transfer_cash_out",
        "score_mismatch:transfer_cash_out",
        "evidence_mismatch:transfer_cash_out",
    }
    mismatch_set = set(mismatch_codes)
    actual = mismatch_set & direct_codes
    if transfer is not None and actual:
        explained = set(actual)
        legacy_score_delta = abs(
            (transfer.legacy_score or 0) - (transfer.plugin_score or 0)
        )
        legacy_points = round(legacy_result["fraud_score"] * 100)
        plugin_points = round(plugin_result["fraud_score"] * 100)
        observed_total_delta = abs(legacy_points - plugin_points)
        other_rule_score_mismatch = any(
            item.rule_id != "transfer_cash_out" and not item.score_matches
            for item in comparisons
        )
        if (
            "score_mismatch:transfer_cash_out" in actual
            and observed_total_delta == legacy_score_delta
            and not other_rule_score_mismatch
        ):
            explained.add("total_score_mismatch")
        if "triggered_mismatch:transfer_cash_out" in actual:
            legacy_triggered = legacy_result["triggered_rules"]
            plugin_triggered = plugin_result["triggered_rules"]
            legacy_count = legacy_triggered.count("transfer_cash_out")
            plugin_count = plugin_triggered.count("transfer_cash_out")
            remaining_variants = _remove_one_occurrence_variants(
                legacy_triggered,
                "transfer_cash_out",
            )
            if (
                legacy_count == plugin_count + 1
                and any(
                    Counter(remaining) == Counter(plugin_triggered)
                    for remaining in remaining_variants
                )
            ):
                explained.add("triggered_members_mismatch")
            if (
                legacy_count == plugin_count + 1
                and plugin_triggered in remaining_variants
            ):
                explained.add("triggered_order_mismatch")
        return (
            (KnownDifference.ZERO_AMOUNT_TRANSFER_CASH_OUT,),
            frozenset(explained & mismatch_set),
        )
    return (), frozenset()


def _remove_one_occurrence_variants(
    values: list[str],
    target: str,
) -> tuple[list[str], ...]:
    return tuple(
        values[:position] + values[position + 1:]
        for position, value in enumerate(values)
        if value == target
    )


def _has_zero_amount_pair(frame: pd.DataFrame) -> bool:
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    step = pd.to_numeric(frame["step"], errors="coerce")
    transfer_mask = frame["type"].eq("TRANSFER") & amount.eq(0)
    cashout_mask = frame["type"].eq("CASH_OUT") & amount.eq(0)
    transfers = frame.loc[transfer_mask, ["nameDest"]].copy()
    cashouts = frame.loc[cashout_mask, ["nameOrig"]].copy()
    if transfers.empty or cashouts.empty:
        return False
    transfers["_step_in"] = step.loc[transfer_mask].to_numpy()
    cashouts["_step_out"] = step.loc[cashout_mask].to_numpy()
    pairs = transfers.merge(
        cashouts,
        left_on="nameDest",
        right_on="nameOrig",
        how="inner",
    )
    if pairs.empty:
        return False
    gap = pairs["_step_out"] - pairs["_step_in"]
    return bool((gap.ge(0) & gap.le(24)).any())


__all__ = [
    "ExecutionFailure",
    "KnownDifference",
    "RuleComparison",
    "ShadowComparisonReport",
    "compare_legacy_and_plugin",
]
