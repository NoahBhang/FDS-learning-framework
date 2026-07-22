"""
shared/scoring/risk_scorer.py
============================================================
RiskScorer — 여러 규칙의 판단을 하나의 종합 보고로 묶는다

[왜 규칙이 직접 점수를 합산하지 않는가 — 책임의 분리]
규칙의 책임: "이 유형의 의심이 있는가"를 판단하는 것.
스코어러의 책임: 여러 판단을 종합하여 등급을 매기는 것.
이 둘을 섞으면, 등급 기준을 바꿀 때마다 모든 규칙을 고쳐야 한다.
1부 문서 13장 "책임이 분리되어 있는가"의 실물이 이 파일이다.

[점수 상한을 100으로 자르는 이유]
규칙이 늘어나면 단순 합산은 무한히 커진다.
관재인에게 "위험도 340점"은 해석 불가능하다.
0~100의 고정 스케일이어야 사람의 직관과 연결된다.
"""

from dataclasses import dataclass, field

from shared.rules.base_fraud_rule import RuleResult


@dataclass
class InvestigationReport:
    """
    관재인 검토 보고서 — 이 시스템의 최종 산출물.

    개별 RuleResult(규칙 하나의 판단)와 구분되는 점:
    이것은 "이 사람을 어느 우선순위로 조사할 것인가"라는
    업무 판단으로 이어지는 종합 문서다.
    """
    debtor_id: str
    total_score: int                    # 0~100 종합 위험 점수
    grade: str                          # 고위험 / 중위험 / 저위험 / 정상
    findings: list = field(default_factory=list)   # 의심 판정된 RuleResult들
    summary: str = ""                   # 사람이 읽는 종합 요약


class RiskScorer:
    """규칙 결과 목록 → 종합 보고서 변환기."""

    # 등급 경계. 실무에서는 관재인과의 협의로 조정한다 (설정값이지 진리가 아니다).
    GRADE_BOUNDS = [
        (70, "고위험"),
        (40, "중위험"),
        (1,  "저위험"),
        (0,  "정상"),
    ]

    def aggregate(self, debtor_id: str, results: list[RuleResult]) -> InvestigationReport:
        findings = [r for r in results if r.is_suspicious]

        # 단순 합산 후 100으로 상한. min()이 상한을 강제한다.
        total = min(100, sum(r.risk_score for r in findings))

        grade = next(g for bound, g in self.GRADE_BOUNDS if total >= bound)

        # 종합 요약 문장 — 보고서의 첫 줄에 올라갈 한 문장을 기계가 조립한다.
        if findings:
            types = ", ".join(r.risk_type for r in findings)
            summary = (
                f"총 {len(findings)}개 유형의 의심 정황({types})이 발견되었다. "
                f"종합 위험 점수 {total}점({grade})으로 "
                f"{'우선 조사가 필요하다.' if total >= 70 else '검토가 필요하다.'}"
            )
        else:
            summary = "발견된 의심 정황이 없다."

        return InvestigationReport(
            debtor_id=debtor_id,
            total_score=total,
            grade=grade,
            findings=findings,
            summary=summary,
        )
