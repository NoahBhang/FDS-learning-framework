"""설정 임계값 이상의 고액 거래를 탐지하는 Plugin Rule."""

from datetime import datetime
import math
from numbers import Integral, Real

import pandas as pd
from pandas.api.types import (
    is_bool,
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)

from .base_rule import BaseRule
from .evidence_item import EvidenceItem
from .rule_result import RuleResult


class LargeTransferRule(BaseRule):
    """고정 임계값 이상의 양수 거래를 탐지한다."""

    rule_id = "large_transfer"
    rule_name = "Large Transfer Rule"
    description = (
        "Detects transactions whose amount is greater than or equal to "
        "the configured threshold."
    )

    def __init__(
        self,
        threshold: float,
        score: int,
    ) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise TypeError("threshold는 bool이 아닌 숫자여야 합니다.")

        normalized_threshold = float(threshold)
        if not math.isfinite(normalized_threshold):
            raise ValueError("threshold는 유한한 숫자여야 합니다.")
        if normalized_threshold <= 0:
            raise ValueError("threshold는 0보다 커야 합니다.")

        if isinstance(score, bool) or not isinstance(score, int):
            raise TypeError("score는 bool이 아닌 int여야 합니다.")
        if not 1 <= score <= 100:
            raise ValueError("score는 1 이상 100 이하여야 합니다.")

        self._threshold = normalized_threshold
        self._score = score

    @property
    def threshold(self) -> float:
        """검증된 탐지 임계값."""

        return self._threshold

    @property
    def score(self) -> int:
        """탐지 시 반환할 고정 점수."""

        return self._score

    def evaluate(
        self,
        transactions: pd.DataFrame,
    ) -> RuleResult:
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions는 pandas DataFrame이어야 합니다.")

        required_columns = ("transaction_id", "amount")
        missing_columns = [
            column
            for column in required_columns
            if column not in transactions.columns
        ]
        if missing_columns:
            raise ValueError(
                "필수 컬럼이 누락되었습니다: "
                + ", ".join(missing_columns)
            )

        amounts = transactions["amount"]
        if (
            is_bool_dtype(amounts.dtype)
            or is_complex_dtype(amounts.dtype)
            or not is_numeric_dtype(amounts.dtype)
        ):
            raise TypeError(
                "amount 컬럼은 bool 또는 complex가 아닌 "
                "real numeric dtype이어야 합니다."
            )

        present_amounts = amounts[amounts.notna()]
        if any(not math.isfinite(float(value)) for value in present_amounts):
            raise ValueError("amount 값은 모두 유한한 숫자여야 합니다.")

        detected_mask = (
            amounts.notna()
            & (amounts > 0)
            & (amounts >= self.threshold)
        ).fillna(False)
        detected_rows = transactions.loc[detected_mask]

        evidence = tuple(
            self._evidence_from_row(row)
            for _, row in detected_rows.iterrows()
        )
        count = len(evidence)
        reason = (
            f"Detected {count} transaction(s) with amount >= "
            f"{self.threshold}."
        )

        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=bool(evidence),
            score=self.score if evidence else 0,
            reason=reason,
            evidence=evidence,
        )

    def _evidence_from_row(self, row: pd.Series) -> EvidenceItem:
        transaction_id = row["transaction_id"]
        if _is_missing(transaction_id):
            raise ValueError("탐지된 거래의 transaction_id 값이 필요합니다.")
        if not isinstance(transaction_id, str):
            raise TypeError(
                "탐지된 거래의 transaction_id는 문자열이어야 합니다."
            )
        if not transaction_id.strip():
            raise ValueError(
                "탐지된 거래의 transaction_id는 비어 있을 수 없습니다."
            )

        return EvidenceItem(
            transaction_id=transaction_id,
            source_row_id=_source_row_id(row.get("source_row_id")),
            actor_account=_optional_string(row.get("actor_account")),
            target_account=_optional_string(row.get("target_account")),
            transaction_datetime=_transaction_datetime(
                row.get("transaction_datetime")
            ),
            amount=float(row["amount"]),
            message=(
                f"Transaction amount {float(row['amount'])} is greater "
                f"than or equal to threshold {self.threshold}."
            ),
        )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, bool) else False


def _optional_string(value: object) -> str | None:
    if _is_missing(value):
        return None
    return str(value)


def _source_row_id(value: object) -> str | int | None:
    if _is_missing(value):
        return None
    if is_bool(value):
        raise TypeError("source_row_id는 bool일 수 없습니다.")
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    return str(value)


def _transaction_datetime(value: object) -> datetime | None:
    if _is_missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    raise TypeError("transaction_datetime 값이 datetime이 아닙니다.")
