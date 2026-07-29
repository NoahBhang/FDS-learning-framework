"""Rule 실행 실패를 기록하는 안전한 오류 모델."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuleExecutionError:
    """사용자에게 노출 가능한 최소 Rule 실행 오류 정보."""

    rule_id: str
    rule_name: str
    error_type: str
    message: str

    def __post_init__(self) -> None:
        for field_name in (
            "rule_id",
            "rule_name",
            "error_type",
            "message",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name}는 문자열이어야 합니다.")
            normalized = value.strip()
            if not normalized:
                raise ValueError(
                    f"{field_name}는 비어 있을 수 없습니다."
                )
            object.__setattr__(self, field_name, normalized)
