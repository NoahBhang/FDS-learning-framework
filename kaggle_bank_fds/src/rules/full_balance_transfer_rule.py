"""Canonical TRANSFER 거래의 전액 또는 사실상 전액 이체를 탐지한다."""

from datetime import datetime
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool

from .base_rule import BaseRule
from .evidence_item import EvidenceItem
from .rule_result import RuleResult


class FullBalanceTransferRule(BaseRule):
    """출발 계좌의 거래 전 잔액 대부분을 이체하는 거래를 탐지한다."""

    rule_id = "full_balance_transfer"
    rule_name = "Full Balance Transfer Rule"
    description = (
        "Detects transfer transactions that move at least the configured ratio "
        "of the originating account's pre-transfer balance, indicating a full "
        "or effectively full balance transfer."
    )

    BASE_SCORE = 15
    SCORE_PER_MATCH = 5
    MAX_SCORE = 30
    REQUIRED_COLUMNS = (
        "transaction_id",
        "source_row_id",
        "transaction_datetime",
        "action_type",
        "amount",
        "balance_before",
        "actor_account",
        "target_account",
    )

    def __init__(
        self,
        *,
        minimum_balance_ratio: float = 0.999,
    ) -> None:
        if isinstance(minimum_balance_ratio, (bool, np.bool_)) or not isinstance(
            minimum_balance_ratio, Real
        ):
            raise TypeError(
                "minimum_balance_ratio must be a bool-free real number."
            )

        normalized_ratio = float(minimum_balance_ratio)
        if not math.isfinite(normalized_ratio):
            raise ValueError("minimum_balance_ratio must be finite.")
        if not 0.0 < normalized_ratio <= 1.0:
            raise ValueError(
                "minimum_balance_ratio must be greater than 0 and at most 1."
            )

        self._minimum_balance_ratio = normalized_ratio

    @property
    def minimum_balance_ratio(self) -> float:
        """검증된 최소 잔액 이체 비율."""

        return self._minimum_balance_ratio

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame.")

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in transactions.columns
        ]
        if missing_columns:
            raise ValueError("Missing required columns: " + ", ".join(missing_columns))

        candidate_mask = transactions["action_type"].eq("TRANSFER").fillna(False)
        candidate_positions = np.flatnonzero(candidate_mask.to_numpy(dtype=bool))
        candidates = transactions.iloc[candidate_positions].loc[
            :, self.REQUIRED_COLUMNS
        ].copy()

        self._validate_candidates(candidates)
        if candidates.empty:
            return self._result(())

        detected_mask = (
            candidates["balance_before"].gt(0)
            & candidates["amount"].gt(0)
            & candidates["amount"].ge(
                candidates["balance_before"] * self.minimum_balance_ratio
            )
        ).fillna(False)
        detected = candidates.loc[detected_mask]

        evidence = tuple(
            self._evidence_from_row(row) for _, row in detected.iterrows()
        )
        return self._result(evidence)

    def _validate_candidates(self, candidates: pd.DataFrame) -> None:
        normalized_ids: list[str] = []
        normalized_actors: list[str] = []
        normalized_targets: list[str] = []
        normalized_amounts: list[float] = []
        normalized_balances: list[float] = []

        for row in candidates.itertuples(index=False):
            normalized_ids.append(
                _required_string(row.transaction_id, "transaction_id")
            )
            normalized_actors.append(
                _required_string(row.actor_account, "actor_account")
            )
            normalized_targets.append(
                _required_string(row.target_account, "target_account")
            )
            _source_row_id(row.source_row_id)
            _transaction_datetime(row.transaction_datetime)
            normalized_amounts.append(_finite_real(row.amount, "amount"))
            normalized_balances.append(
                _finite_real(row.balance_before, "balance_before")
            )

        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("Candidate transaction_id values must be unique.")

        candidates["transaction_id"] = normalized_ids
        candidates["actor_account"] = normalized_actors
        candidates["target_account"] = normalized_targets
        candidates["amount"] = normalized_amounts
        candidates["balance_before"] = normalized_balances

    def _evidence_from_row(self, row: pd.Series) -> EvidenceItem:
        amount = float(row["amount"])
        balance_before = float(row["balance_before"])
        actual_ratio = amount / balance_before
        return EvidenceItem(
            transaction_id=row["transaction_id"],
            source_row_id=_source_row_id(row["source_row_id"]),
            actor_account=row["actor_account"],
            target_account=row["target_account"],
            transaction_datetime=_transaction_datetime(
                row["transaction_datetime"]
            ),
            amount=amount,
            message=(
                f"Transferred {actual_ratio:.2%} of pre-transfer balance "
                f"{balance_before:.2f}; configured minimum is "
                f"{self.minimum_balance_ratio:.2%}."
            ),
        )

    def _result(self, evidence: tuple[EvidenceItem, ...]) -> RuleResult:
        match_count = len(evidence)
        reason = (
            f"Detected {match_count} full or effectively full balance "
            "transfer(s) at or above the configured minimum ratio of "
            f"{self.minimum_balance_ratio:.2%}."
        )
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=bool(evidence),
            score=(
                min(
                    self.MAX_SCORE,
                    self.BASE_SCORE + self.SCORE_PER_MATCH * match_count,
                )
                if evidence
                else 0
            ),
            reason=reason,
            evidence=evidence,
        )


def _required_string(value: object, field_name: str) -> str:
    if _is_missing(value):
        raise ValueError(f"Candidate {field_name} must not be missing.")
    if not isinstance(value, str):
        raise TypeError(f"Candidate {field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"Candidate {field_name} must not be empty.")
    return normalized


def _finite_real(value: object, field_name: str) -> float:
    if _is_missing(value):
        raise ValueError(f"Candidate {field_name} must not be missing.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"Candidate {field_name} must be a bool-free real number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"Candidate {field_name} must be finite.")
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
