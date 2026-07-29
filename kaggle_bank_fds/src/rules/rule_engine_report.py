"""Rule Engine 전체 실행 결과 모델."""

from dataclasses import dataclass

from .rule_execution_error import RuleExecutionError
from .rule_result import RuleResult


@dataclass(frozen=True, slots=True)
class RuleEngineReport:
    """성공 결과와 실행 오류를 분리해 보관한다."""

    results: tuple[RuleResult, ...] = ()
    errors: tuple[RuleExecutionError, ...] = ()

    def __post_init__(self) -> None:
        normalized_results = tuple(self.results)
        normalized_errors = tuple(self.errors)

        if not all(
            isinstance(result, RuleResult)
            for result in normalized_results
        ):
            raise TypeError("results의 모든 원소는 RuleResult여야 합니다.")
        if not all(
            isinstance(error, RuleExecutionError)
            for error in normalized_errors
        ):
            raise TypeError(
                "errors의 모든 원소는 RuleExecutionError여야 합니다."
            )

        object.__setattr__(self, "results", normalized_results)
        object.__setattr__(self, "errors", normalized_errors)

    @property
    def succeeded_count(self) -> int:
        return len(self.results)

    @property
    def failed_count(self) -> int:
        return len(self.errors)

    @property
    def triggered_results(self) -> tuple[RuleResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.triggered
        )
