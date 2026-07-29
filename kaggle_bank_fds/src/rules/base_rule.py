"""은행 FDS Plugin Rule이 따라야 하는 공통 인터페이스."""

from abc import ABC, abstractmethod

import pandas as pd

from .rule_result import RuleResult


class BaseRule(ABC):
    """Rule Engine에 등록할 모든 탐지 Rule의 추상 기반 클래스."""

    rule_id: str
    rule_name: str
    description: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)

        evaluate_method = getattr(cls, "evaluate", None)
        if getattr(evaluate_method, "__isabstractmethod__", False):
            return

        validate_rule_metadata(cls)

    @abstractmethod
    def evaluate(
        self,
        transactions: pd.DataFrame,
    ) -> RuleResult:
        """거래 DataFrame을 평가하고 단일 RuleResult를 반환한다."""

        raise NotImplementedError


def validate_rule_metadata(rule: object) -> None:
    """구체 Rule의 필수 클래스 메타데이터를 검증한다."""

    for field_name in ("rule_id", "rule_name", "description"):
        try:
            value = getattr(rule, field_name)
        except AttributeError as error:
            raise TypeError(
                f"구체 Rule은 {field_name} 메타데이터를 제공해야 합니다."
            ) from error

        if not isinstance(value, str):
            raise TypeError(f"{field_name}는 문자열이어야 합니다.")
        if not value.strip():
            raise ValueError(f"{field_name}는 비어 있을 수 없습니다.")
