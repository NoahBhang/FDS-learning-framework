"""Detect threshold-avoidance transaction splitting by the same sender."""

from datetime import datetime
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool

from .base_rule import BaseRule
from .evidence_item import EvidenceItem
from .rule_result import RuleResult


class SplitTransactionRule(BaseRule):
    rule_id = "split_transaction"
    rule_name = "Split Transaction Rule"
    description = (
        "Detects multiple sub-ceiling transfers from one sender whose "
        "cumulative amount reaches a configured threshold within a step window."
    )
    REQUIRED_COLUMNS = (
        "transaction_id", "source_row_id", "step", "transaction_datetime",
        "action_type", "amount", "actor_account", "target_account",
    )

    def __init__(self, *, individual_amount_ceiling: float = 100_000.0,
                 cumulative_amount_threshold: float = 200_000.0,
                 max_step_gap: int = 24, min_count: int = 3,
                 score: int = 35) -> None:
        ceiling = _positive_real(individual_amount_ceiling, "individual_amount_ceiling")
        cumulative = _positive_real(cumulative_amount_threshold, "cumulative_amount_threshold")
        if cumulative <= ceiling:
            raise ValueError(
                "cumulative_amount_threshold must be greater than individual_amount_ceiling."
            )
        self._individual_amount_ceiling = ceiling
        self._cumulative_amount_threshold = cumulative
        self._max_step_gap = _nonnegative_integer(max_step_gap, "max_step_gap")
        self._min_count = _minimum_count(min_count)
        self._score = _score(score)

    @property
    def individual_amount_ceiling(self) -> float: return self._individual_amount_ceiling

    @property
    def cumulative_amount_threshold(self) -> float: return self._cumulative_amount_threshold

    @property
    def max_step_gap(self) -> int: return self._max_step_gap

    @property
    def min_count(self) -> int: return self._min_count

    @property
    def score(self) -> int: return self._score

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        candidates = _candidate_frame(transactions, self.REQUIRED_COLUMNS)
        eligible = candidates.loc[
            candidates["amount"].gt(0)
            & candidates["amount"].lt(self.individual_amount_ceiling)
        ].copy()
        detected_positions: set[int] = set()
        for _, group in eligible.groupby("actor_account", sort=False, dropna=False):
            detected_positions.update(
                _qualifying_positions(
                    group, self.max_step_gap, self.min_count,
                    self.cumulative_amount_threshold,
                )
            )
        detected = candidates.loc[
            candidates["_input_position"].isin(detected_positions)
        ].sort_values("_input_position", kind="stable")
        recipient_count = detected["target_account"].nunique() if not detected.empty else 0
        message = (
            f"Transaction belongs to a sub-ceiling split window with at least "
            f"{self.min_count} transaction(s), cumulative threshold "
            f"{self.cumulative_amount_threshold:.2f}, and individual ceiling "
            f"{self.individual_amount_ceiling:.2f}."
        )
        evidence = tuple(_evidence(row, message) for _, row in detected.iterrows())
        return self._result(evidence, int(recipient_count))

    def _result(self, evidence: tuple[EvidenceItem, ...], recipient_count: int) -> RuleResult:
        return RuleResult(
            rule_id=self.rule_id, rule_name=self.rule_name,
            triggered=bool(evidence), score=self.score if evidence else 0,
            reason=(
                f"Detected {len(evidence)} sub-ceiling transfer(s) across "
                f"{recipient_count} recipient(s) within {self.max_step_gap} "
                f"step(s), cumulative threshold {self.cumulative_amount_threshold:.2f}, "
                f"and individual ceiling {self.individual_amount_ceiling:.2f}."
            ), evidence=evidence,
        )


def _qualifying_positions(group, max_gap, min_count, min_total) -> set[int]:
    ordered = group.sort_values(["step", "_input_position"], kind="stable")
    steps = ordered["step"].to_numpy(dtype=np.int64)
    amounts = ordered["amount"].to_numpy(dtype=float)
    positions = ordered["_input_position"].to_numpy(dtype=np.int64)
    marks = np.zeros(len(ordered) + 1, dtype=np.int64)
    left, total = 0, 0.0
    for right in range(len(ordered)):
        total += amounts[right]
        while steps[right] - steps[left] > max_gap:
            total -= amounts[left]
            left += 1
        if right - left + 1 >= min_count and total >= min_total:
            marks[left] += 1
            marks[right + 1] -= 1
    return {
        int(position)
        for position in positions[np.cumsum(marks[:-1]) > 0]
    }


def _candidate_frame(transactions, required_columns) -> pd.DataFrame:
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame.")
    missing = [column for column in required_columns if column not in transactions.columns]
    if missing: raise ValueError("Missing required columns: " + ", ".join(missing))
    mask = transactions["action_type"].eq("TRANSFER").fillna(False)
    positions = np.flatnonzero(mask.to_numpy(dtype=bool))
    candidates = transactions.iloc[positions].loc[:, required_columns].copy()
    candidates["_input_position"] = positions
    _validate_candidates(candidates)
    return candidates


def _validate_candidates(candidates: pd.DataFrame) -> None:
    if isinstance(candidates["step"].dtype, pd.CategoricalDtype):
        raise TypeError("Candidate step must not use categorical dtype.")
    if isinstance(candidates["amount"].dtype, pd.CategoricalDtype):
        raise TypeError("Candidate amount must not use categorical dtype.")
    ids, actors, targets, steps, amounts = [], [], [], [], []
    for row in candidates.itertuples(index=False):
        ids.append(_required_string(row.transaction_id, "transaction_id"))
        actors.append(_required_string(row.actor_account, "actor_account"))
        targets.append(_required_string(row.target_account, "target_account"))
        _source_row_id(row.source_row_id)
        steps.append(_step(row.step)); amounts.append(_amount(row.amount))
        _transaction_datetime(row.transaction_datetime)
    if len(ids) != len(set(ids)): raise ValueError("Candidate transaction_id values must be unique.")
    candidates["transaction_id"], candidates["actor_account"] = ids, actors
    candidates["target_account"], candidates["step"] = targets, steps
    candidates["amount"] = amounts


def _integer(value, name) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a bool-free integer.")
    return int(value)


def _nonnegative_integer(value, name) -> int:
    normalized = _integer(value, name)
    if normalized < 0: raise ValueError(f"{name} must be greater than or equal to 0.")
    return normalized


def _minimum_count(value) -> int:
    normalized = _integer(value, "min_count")
    if normalized < 2: raise ValueError("min_count must be at least 2.")
    return normalized


def _positive_real(value, name) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a bool-free real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be finite and greater than 0.")
    return normalized


def _score(value) -> int:
    normalized = _integer(value, "score")
    if not 1 <= normalized <= 100: raise ValueError("score must be between 1 and 100.")
    return normalized


def _required_string(value, name) -> str:
    if _is_missing(value): raise ValueError(f"Candidate {name} must not be missing.")
    if not isinstance(value, str): raise TypeError(f"Candidate {name} must be a string.")
    normalized = value.strip()
    if not normalized: raise ValueError(f"Candidate {name} must not be empty.")
    return normalized


def _step(value) -> int:
    if _is_missing(value): raise ValueError("Candidate step must not be missing.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("Candidate step must be a bool-free integer.")
    return int(value)


def _amount(value) -> float:
    if _is_missing(value): raise ValueError("Candidate amount must not be missing.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("Candidate amount must be a bool-free real number.")
    normalized = float(value)
    if not math.isfinite(normalized): raise ValueError("Candidate amount must be finite.")
    return normalized


def _source_row_id(value):
    if _is_missing(value): return None
    if is_bool(value): raise TypeError("Candidate source_row_id must not be bool.")
    if isinstance(value, Integral): return int(value)
    if isinstance(value, str): return value.strip()
    raise TypeError("Candidate source_row_id must be str, int, or missing.")


def _transaction_datetime(value):
    if _is_missing(value): return None
    if isinstance(value, pd.Timestamp): return value.to_pydatetime()
    if isinstance(value, datetime): return value
    raise TypeError("Candidate transaction_datetime must be datetime or missing.")


def _is_missing(value) -> bool:
    if value is None or value is pd.NA or value is pd.NaT: return True
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _evidence(row, message) -> EvidenceItem:
    return EvidenceItem(
        transaction_id=row["transaction_id"], source_row_id=_source_row_id(row["source_row_id"]),
        actor_account=row["actor_account"], target_account=row["target_account"],
        transaction_datetime=_transaction_datetime(row["transaction_datetime"]),
        amount=row["amount"], message=message,
    )
