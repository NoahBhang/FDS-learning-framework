"""Canonical 거래의 TRANSFER 이후 CASH_OUT 연쇄를 탐지하는 Plugin Rule."""

from datetime import datetime
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool

from .base_rule import BaseRule
from .evidence_item import EvidenceItem
from .rule_result import RuleResult


class TransferCashOutRule(BaseRule):
    """수취 계좌에서 이어지는 유사 금액 cash-out pair를 탐지한다."""

    rule_id = "transfer_cash_out"
    rule_name = "Transfer Cash-Out Rule"
    description = (
        "Detects transfers whose receiving account performs a subsequent "
        "cash-out within the configured step window and amount tolerance."
    )

    BASE_SCORE = 20
    SCORE_PER_PAIR = 5
    MAX_SCORE = 40
    REQUIRED_COLUMNS = (
        "transaction_id",
        "source_row_id",
        "step",
        "transaction_datetime",
        "action_type",
        "amount",
        "actor_account",
        "target_account",
    )

    def __init__(
        self,
        *,
        max_step_gap: int = 24,
        amount_tolerance: float = 0.05,
    ) -> None:
        if isinstance(max_step_gap, (bool, np.bool_)) or not isinstance(
            max_step_gap, Integral
        ):
            raise TypeError("max_step_gap must be a bool-free integer.")
        normalized_gap = int(max_step_gap)
        if normalized_gap < 0:
            raise ValueError("max_step_gap must be greater than or equal to 0.")

        if isinstance(amount_tolerance, (bool, np.bool_)) or not isinstance(
            amount_tolerance, Real
        ):
            raise TypeError("amount_tolerance must be a bool-free real number.")
        normalized_tolerance = float(amount_tolerance)
        if not math.isfinite(normalized_tolerance):
            raise ValueError("amount_tolerance must be finite.")
        if not 0.0 <= normalized_tolerance <= 1.0:
            raise ValueError("amount_tolerance must be between 0 and 1.")

        self._max_step_gap = normalized_gap
        self._amount_tolerance = normalized_tolerance

    @property
    def max_step_gap(self) -> int:
        """검증된 최대 step 간격."""

        return self._max_step_gap

    @property
    def amount_tolerance(self) -> float:
        """TRANSFER 금액을 기준으로 한 허용 오차 비율."""

        return self._amount_tolerance

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame.")

        missing_columns = [
            column for column in self.REQUIRED_COLUMNS if column not in transactions.columns
        ]
        if missing_columns:
            raise ValueError("Missing required columns: " + ", ".join(missing_columns))

        candidate_mask = transactions["action_type"].isin(("TRANSFER", "CASH_OUT"))
        candidate_positions = np.flatnonzero(candidate_mask.to_numpy(dtype=bool))
        candidates = transactions.iloc[candidate_positions].loc[
            :, self.REQUIRED_COLUMNS
        ].copy()
        candidates["_input_position"] = candidate_positions

        self._validate_candidates(candidates)
        if candidates.empty:
            return self._result(())

        positive_candidates = candidates.loc[candidates["amount"].gt(0)].copy()
        transfers = positive_candidates.loc[
            positive_candidates["action_type"].eq("TRANSFER")
        ]
        cashouts = positive_candidates.loc[
            positive_candidates["action_type"].eq("CASH_OUT")
        ]
        if transfers.empty or cashouts.empty:
            return self._result(())

        transfer_side = _prefixed_side(transfers, "transfer")
        cashout_side = _prefixed_side(cashouts, "cashout")
        merged = transfer_side.merge(
            cashout_side,
            left_on="transfer_target_account",
            right_on="cashout_actor_account",
            how="inner",
            sort=False,
        )
        if merged.empty:
            return self._result(())

        gap = merged["cashout_step"] - merged["transfer_step"]
        amount_difference = (
            merged["cashout_amount"] - merged["transfer_amount"]
        ).abs()
        matched = merged.loc[
            gap.ge(0)
            & gap.le(self.max_step_gap)
            & amount_difference.le(
                merged["transfer_amount"] * self.amount_tolerance
            )
        ].copy()
        if matched.empty:
            return self._result(())

        matched.sort_values(
            ["transfer__input_position", "cashout__input_position"],
            kind="stable",
            inplace=True,
        )
        unique_rows = []
        seen_pairs: set[tuple[str, str]] = set()
        for _, row in matched.iterrows():
            identity = (
                row["transfer_transaction_id"],
                row["cashout_transaction_id"],
            )
            if identity not in seen_pairs:
                seen_pairs.add(identity)
                unique_rows.append(row)

        evidence: list[EvidenceItem] = []
        for row in unique_rows:
            transfer_id = row["transfer_transaction_id"]
            cashout_id = row["cashout_transaction_id"]
            evidence.append(
                _evidence_from_pair_row(
                    row,
                    prefix="transfer",
                    message=(
                        f"Transfer matched cash-out transaction {cashout_id}."
                    ),
                )
            )
            evidence.append(
                _evidence_from_pair_row(
                    row,
                    prefix="cashout",
                    message=(
                        f"Cash-out matched transfer transaction {transfer_id}."
                    ),
                )
            )

        return self._result(tuple(evidence))

    def _validate_candidates(self, candidates: pd.DataFrame) -> None:
        if isinstance(candidates["step"].dtype, pd.CategoricalDtype):
            raise TypeError("Candidate step must not use categorical dtype.")
        if isinstance(candidates["amount"].dtype, pd.CategoricalDtype):
            raise TypeError("Candidate amount must not use categorical dtype.")

        normalized_ids: list[str] = []
        normalized_actors: list[str] = []
        normalized_targets: list[str] = []

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
            _step(row.step)
            _amount(row.amount)
            _transaction_datetime(row.transaction_datetime)

        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("Candidate transaction_id values must be unique.")

        candidates["transaction_id"] = normalized_ids
        candidates["actor_account"] = normalized_actors
        candidates["target_account"] = normalized_targets
        candidates["step"] = [_step(value) for value in candidates["step"]]
        candidates["amount"] = [_amount(value) for value in candidates["amount"]]

    def _result(self, evidence: tuple[EvidenceItem, ...]) -> RuleResult:
        pair_count = len(evidence) // 2
        unique_transaction_count = len(
            {item.transaction_id for item in evidence}
        )
        reason = (
            f"Detected {pair_count} transfer-to-cash-out pair(s) within "
            f"{self.max_step_gap} step(s) and "
            f"{self.amount_tolerance:.2%} amount tolerance; "
            f"{unique_transaction_count} unique transaction(s) involved."
        )
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=bool(evidence),
            score=(
                min(
                    self.MAX_SCORE,
                    self.BASE_SCORE + self.SCORE_PER_PAIR * pair_count,
                )
                if evidence
                else 0
            ),
            reason=reason,
            evidence=evidence,
        )


def _prefixed_side(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.rename(
        columns={column: f"{prefix}_{column}" for column in frame.columns}
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


def _step(value: object) -> int:
    if _is_missing(value):
        raise ValueError("Candidate step must not be missing.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("Candidate step must be a bool-free integer.")
    return int(value)


def _amount(value: object) -> float:
    if _is_missing(value):
        raise ValueError("Candidate amount must not be missing.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("Candidate amount must be a bool-free real number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("Candidate amount must be finite.")
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


def _evidence_from_pair_row(
    row: pd.Series,
    *,
    prefix: str,
    message: str,
) -> EvidenceItem:
    return EvidenceItem(
        transaction_id=row[f"{prefix}_transaction_id"],
        source_row_id=_source_row_id(row[f"{prefix}_source_row_id"]),
        actor_account=row[f"{prefix}_actor_account"],
        target_account=row[f"{prefix}_target_account"],
        transaction_datetime=_transaction_datetime(
            row[f"{prefix}_transaction_datetime"]
        ),
        amount=row[f"{prefix}_amount"],
        message=message,
    )


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False
