"""은행 FDS Rule 실행 결과 모델."""

from dataclasses import dataclass

from .evidence_item import EvidenceItem


@dataclass(frozen=True, slots=True)
class RuleResult:
    """하나의 Rule이 반환하는 표준 평가 결과."""

    rule_id: str
    rule_name: str
    triggered: bool
    score: int
    reason: str
    evidence: tuple[EvidenceItem, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("rule_id", "rule_name", "reason"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name}는 문자열이어야 합니다.")
            normalized = value.strip()
            if not normalized:
                raise ValueError(
                    f"{field_name}는 비어 있을 수 없습니다."
                )
            object.__setattr__(self, field_name, normalized)

        if not isinstance(self.triggered, bool):
            raise TypeError("triggered는 bool이어야 합니다.")
        if isinstance(self.score, bool) or not isinstance(self.score, int):
            raise TypeError("score는 bool이 아닌 int여야 합니다.")
        if not 0 <= self.score <= 100:
            raise ValueError("score는 0 이상 100 이하여야 합니다.")

        normalized_evidence = tuple(self.evidence)
        if not all(
            isinstance(item, EvidenceItem)
            for item in normalized_evidence
        ):
            raise TypeError(
                "evidence의 모든 원소는 EvidenceItem이어야 합니다."
            )

        if self.triggered:
            if self.score < 1:
                raise ValueError(
                    "triggered=True이면 score는 1 이상이어야 합니다."
                )
            if not normalized_evidence:
                raise ValueError(
                    "triggered=True이면 evidence가 필요합니다."
                )
        else:
            if self.score != 0:
                raise ValueError(
                    "triggered=False이면 score는 0이어야 합니다."
                )
            if normalized_evidence:
                raise ValueError(
                    "triggered=False이면 evidence는 비어 있어야 합니다."
                )

        object.__setattr__(self, "evidence", normalized_evidence)
