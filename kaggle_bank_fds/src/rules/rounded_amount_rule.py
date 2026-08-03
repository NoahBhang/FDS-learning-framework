"""Detect high-value transactions aligned to an exact integer currency unit."""

from datetime import datetime
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool

from .base_rule import BaseRule
from .evidence_item import EvidenceItem
from .rule_result import RuleResult


class RoundedAmountRule(BaseRule):
    rule_id = "rounded_amount"
    rule_name = "Rounded Amount Rule"
    description = (
        "Detects high-value transfer or cash-out transactions aligned to an "
        "exact configured integer currency unit."
    )
    REQUIRED_COLUMNS = (
        "transaction_id", "source_row_id", "step", "transaction_datetime",
        "action_type", "amount", "actor_account", "target_account",
    )

    def __init__(
        self,
        *,
        min_amount: float = 100_000.0,
        rounding_unit: int = 10_000,
        score: int = 20,
    ) -> None:
        normalized_min = _positive_real(min_amount, "min_amount")
        normalized_unit = _positive_integer(rounding_unit, "rounding_unit")
        normalized_score = _score(score)
        if normalized_min < normalized_unit:
            raise ValueError("min_amount must be greater than or equal to rounding_unit.")
        self._min_amount = normalized_min
        self._rounding_unit = normalized_unit
        self._score = normalized_score

    @property
    def min_amount(self) -> float:
        return self._min_amount

    @property
    def rounding_unit(self) -> int:
        return self._rounding_unit

    @property
    def score(self) -> int:
        return self._score

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        candidates = _candidate_frame(
            transactions,
            self.REQUIRED_COLUMNS,
            ("TRANSFER", "CASH_OUT"),
        )
        if candidates.empty:
            return self._result(())

        detected_mask = candidates["amount"].map(
            lambda value: _is_rounded_amount(
                value,
                min_amount=self.min_amount,
                rounding_unit=self.rounding_unit,
            )
        )
        detected = candidates.loc[detected_mask].copy().sort_values(
            "_input_position", kind="stable"
        )
        detected["amount"] = [
            _evidence_amount(value) for value in detected["amount"]
        ]
        evidence = tuple(_evidence(row, self._message()) for _, row in detected.iterrows())
        return self._result(evidence)

    def _message(self) -> str:
        return (
            f"Amount meets minimum {self.min_amount:.2f} and is aligned to "
            f"integer currency unit {self.rounding_unit}."
        )

    def _result(self, evidence: tuple[EvidenceItem, ...]) -> RuleResult:
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=bool(evidence),
            score=self.score if evidence else 0,
            reason=(
                f"Detected {len(evidence)} transaction(s) at or above "
                f"{self.min_amount:.2f} aligned to rounding unit "
                f"{self.rounding_unit}."
            ),
            evidence=evidence,
        )


def _candidate_frame(transactions, required_columns, actions) -> pd.DataFrame:
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame.")
    missing = [column for column in required_columns if column not in transactions.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    mask = transactions["action_type"].isin(actions)
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
        steps.append(_step(row.step))
        amounts.append(_amount(row.amount))
        _transaction_datetime(row.transaction_datetime)
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate transaction_id values must be unique.")
    candidates["transaction_id"] = ids
    candidates["actor_account"] = actors
    candidates["target_account"] = targets
    candidates["step"] = steps
    candidates["amount"] = pd.Series(
        amounts,
        index=candidates.index,
        dtype=object,
    )


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a bool-free real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be finite and greater than 0.")
    return normalized


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a bool-free integer.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return normalized


def _score(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("score must be a bool-free integer.")
    normalized = int(value)
    if not 1 <= normalized <= 100:
        raise ValueError("score must be between 1 and 100.")
    return normalized


def _required_string(value: object, name: str) -> str:
    if _is_missing(value):
        raise ValueError(f"Candidate {name} must not be missing.")
    if not isinstance(value, str):
        raise TypeError(f"Candidate {name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"Candidate {name} must not be empty.")
    return normalized


def _step(value: object) -> int:
    if _is_missing(value):
        raise ValueError("Candidate step must not be missing.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("Candidate step must be a bool-free integer.")
    return int(value)


def _amount(value: object) -> int | float:
    if _is_missing(value):
        raise ValueError("Candidate amount must not be missing.")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Candidate amount must be a bool-free real number.")
    if isinstance(value, Integral):
        return int(value)
    if not isinstance(value, Real):
        raise TypeError("Candidate amount must be a bool-free real number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("Candidate amount must be finite.")
    return normalized


def _is_rounded_amount(
    value: int | float,
    *,
    min_amount: float,
    rounding_unit: int,
) -> bool:
    if value <= 0 or value < min_amount:
        return False
    if isinstance(value, int):
        return value % rounding_unit == 0
    return value.is_integer() and int(value) % rounding_unit == 0


def _evidence_amount(value: int | float) -> float:
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(
            "Candidate amount is outside the supported finite float range."
        ) from error
    if not math.isfinite(normalized):
        raise ValueError(
            "Candidate amount is outside the supported finite float range."
        )
    return normalized


def _source_row_id(value: object) -> str | int | None:
    if _is_missing(value):
        return None
    if is_bool(value):
        raise TypeError("Candidate source_row_id must not be bool.")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        return value.strip()
    raise TypeError("Candidate source_row_id must be str, int, or missing.")


def _transaction_datetime(value: object) -> datetime | None:
    if _is_missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    raise TypeError("Candidate transaction_datetime must be datetime or missing.")


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _evidence(row: pd.Series, message: str) -> EvidenceItem:
    return EvidenceItem(
        transaction_id=row["transaction_id"],
        source_row_id=_source_row_id(row["source_row_id"]),
        actor_account=row["actor_account"],
        target_account=row["target_account"],
        transaction_datetime=_transaction_datetime(row["transaction_datetime"]),
        amount=row["amount"],
        message=message,
    )
