"""Rule Engine Framework 검증용 Dummy Rule."""

from datetime import datetime
from numbers import Integral, Real

import pandas as pd

from .base_rule import BaseRule
from .evidence_item import EvidenceItem
from .rule_result import RuleResult


class AlwaysTrueRule(BaseRule):
    """입력과 관계없이 항상 탐지되는 테스트 Rule."""

    rule_id = "always_true"
    rule_name = "Always True Rule"
    description = "Rule Engine의 triggered 경로를 검증한다."

    def evaluate(
        self,
        transactions: pd.DataFrame,
    ) -> RuleResult:
        if "transaction_id" not in transactions.columns:
            raise ValueError("transaction_id 컬럼이 필요합니다.")

        evidence = tuple(
            _evidence_from_row(row)
            for _, row in transactions.iterrows()
        )
        if not evidence:
            raise ValueError(
                "AlwaysTrueRule은 최소 한 건의 거래가 필요합니다."
            )

        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=True,
            score=100,
            reason="테스트를 위해 항상 탐지됩니다.",
            evidence=evidence,
        )


class AlwaysFalseRule(BaseRule):
    """입력과 관계없이 절대 탐지되지 않는 테스트 Rule."""

    rule_id = "always_false"
    rule_name = "Always False Rule"
    description = "Rule Engine의 non-triggered 경로를 검증한다."

    def evaluate(
        self,
        transactions: pd.DataFrame,
    ) -> RuleResult:
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=False,
            score=0,
            reason="테스트를 위해 탐지되지 않습니다.",
            evidence=(),
        )


def _evidence_from_row(row: pd.Series) -> EvidenceItem:
    return EvidenceItem(
        transaction_id=_required_value(row, "transaction_id"),
        source_row_id=_source_row_id(row.get("source_row_id")),
        actor_account=_optional_string(row.get("actor_account")),
        target_account=_optional_string(row.get("target_account")),
        transaction_datetime=_transaction_datetime(
            row.get("transaction_datetime")
        ),
        amount=_amount(row.get("amount")),
        message="AlwaysTrueRule 테스트 근거",
    )


def _required_value(row: pd.Series, column: str) -> object:
    value = row[column]
    if pd.isna(value):
        raise ValueError(f"{column} 값이 필요합니다.")
    return value


def _optional_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _source_row_id(value: object) -> str | int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    return str(value)


def _transaction_datetime(value: object) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    raise TypeError("transaction_datetime 값이 datetime이 아닙니다.")


def _amount(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    raise TypeError("amount 값이 숫자가 아닙니다.")
