"""등록된 Plugin Rule을 실행하는 은행 FDS Rule Engine."""

import pandas as pd

from .rule_registry import RuleRegistry
from .rule_engine_report import RuleEngineReport
from .rule_execution_error import RuleExecutionError
from .rule_result import RuleResult


class RuleEngine:
    """RuleRegistry의 모든 Rule을 등록 순서대로 실행한다."""

    def __init__(self, registry: RuleRegistry) -> None:
        if not isinstance(registry, RuleRegistry):
            raise TypeError("registry는 RuleRegistry여야 합니다.")

        self.registry = registry

    def evaluate(
        self,
        transactions: pd.DataFrame,
    ) -> RuleEngineReport:
        """모든 Rule을 실행하고 성공 결과와 오류를 분리해 반환한다."""

        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions는 pandas DataFrame이어야 합니다.")

        results: list[RuleResult] = []
        errors: list[RuleExecutionError] = []

        for rule in self.registry.get_rules():
            try:
                result = rule.evaluate(transactions)
                if not isinstance(result, RuleResult):
                    raise TypeError(
                        "Rule evaluate()는 RuleResult를 반환해야 합니다."
                    )
                results.append(result)
            except Exception as error:
                errors.append(
                    RuleExecutionError(
                        rule_id=rule.rule_id,
                        rule_name=rule.rule_name,
                        error_type=type(error).__name__,
                        message=str(error) or "메시지 없는 Rule 실행 오류",
                    )
                )

        return RuleEngineReport(
            results=tuple(results),
            errors=tuple(errors),
        )
