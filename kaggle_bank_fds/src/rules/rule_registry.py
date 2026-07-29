"""은행 FDS Plugin Rule 등록소."""

from .base_rule import BaseRule, validate_rule_metadata


class RuleRegistry:
    """Rule을 등록 순서대로 보관하고 조회한다."""

    def __init__(self) -> None:
        self._rules: dict[str, BaseRule] = {}

    def register(self, rule: BaseRule) -> None:
        """Rule을 등록하고 중복 rule_id를 거부한다."""

        if not isinstance(rule, BaseRule):
            raise TypeError("BaseRule 인스턴스만 등록할 수 있습니다.")

        validate_rule_metadata(rule)

        if rule.rule_id != rule.rule_id.strip():
            raise ValueError(
                "rule_id 앞뒤에는 공백을 사용할 수 없습니다."
            )

        if rule.rule_id in self._rules:
            raise ValueError(
                f"이미 등록된 rule_id입니다: {rule.rule_id}"
            )

        self._rules[rule.rule_id] = rule

    def get(self, rule_id: str) -> BaseRule:
        """rule_id로 등록된 Rule을 조회한다."""

        try:
            return self._rules[rule_id]
        except KeyError as error:
            raise KeyError(
                f"등록되지 않은 rule_id입니다: {rule_id}"
            ) from error

    def get_rules(self) -> list[BaseRule]:
        """실행할 Rule을 등록 순서대로 반환한다."""

        return list(self._rules.values())
