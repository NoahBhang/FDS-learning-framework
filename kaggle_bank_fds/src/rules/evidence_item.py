"""RuleResult에 포함되는 불변 거래 근거 모델."""

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real

import pandas as pd


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """의심 판단을 뒷받침하는 단일 거래 근거."""

    transaction_id: str
    source_row_id: str | int | None
    actor_account: str | None
    target_account: str | None
    transaction_datetime: datetime | None
    amount: float | None
    message: str

    def __post_init__(self) -> None:
        transaction_id = _required_string(
            self.transaction_id,
            "transaction_id",
        )
        message = _required_string(self.message, "message")

        object.__setattr__(
            self,
            "transaction_id",
            transaction_id,
        )
        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "actor_account",
            _optional_string(self.actor_account, "actor_account"),
        )
        object.__setattr__(
            self,
            "target_account",
            _optional_string(self.target_account, "target_account"),
        )

        source_row_id = self.source_row_id
        if isinstance(source_row_id, bool) or not isinstance(
            source_row_id,
            (str, int, type(None)),
        ):
            raise TypeError(
                "source_row_id는 str, int 또는 None이어야 합니다."
            )
        if isinstance(source_row_id, str):
            object.__setattr__(
                self,
                "source_row_id",
                source_row_id.strip(),
            )

        transaction_datetime = self.transaction_datetime
        if transaction_datetime is None or transaction_datetime is pd.NaT:
            object.__setattr__(self, "transaction_datetime", None)
        elif not isinstance(transaction_datetime, datetime):
            raise TypeError(
                "transaction_datetime은 datetime 또는 None이어야 합니다."
            )

        if self.amount is not None:
            if isinstance(self.amount, bool) or not isinstance(
                self.amount,
                Real,
            ):
                raise TypeError("amount는 유한한 숫자 또는 None이어야 합니다.")
            normalized_amount = float(self.amount)
            if not math.isfinite(normalized_amount):
                raise ValueError("amount는 유한한 숫자여야 합니다.")
            object.__setattr__(self, "amount", normalized_amount)


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}는 문자열이어야 합니다.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name}는 비어 있을 수 없습니다.")

    return normalized


def _optional_string(
    value: object,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name}는 문자열 또는 None이어야 합니다.")
    return value.strip()
