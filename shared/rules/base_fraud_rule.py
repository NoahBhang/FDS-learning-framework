"""
shared/rules/base_fraud_rule.py
============================================================
BaseFraudRule — 모든 의심 규칙의 조상

[이 파일이 전체 시스템에서 가장 중요한 이유]
FDS의 본질은 "규칙은 계속 늘어난다"는 것이다.
쪼개기 송금 → 편파 변제 → 레이어링 → (미래의 새 수법)...
새 규칙이 추가될 때마다 파이프라인 코드를 고쳐야 한다면
그 시스템은 유지보수 불가능해진다.

해법이 추상 클래스다. 파이프라인은 오직 이 약속만 안다:
    "모든 규칙은 evaluate(df)를 가지며 RuleResult를 돌려준다"
새 규칙은 이 약속만 지키면 파이프라인 수정 없이 꽂힌다.
→ 개방-폐쇄 원칙(OCP): 확장에는 열려 있고, 수정에는 닫혀 있다.

은행권 FDS와 파산관재인 FDS가 이 파일 하나를 공유하는 것이
"같은 사고틀, 다른 도메인" 전략의 물리적 실체다.

[evaluate가 DataFrame을 받는 이유]
DB가 장기 기억이라면 DataFrame은 작업 기억(working memory)이다.
규칙은 집계·필터링·그룹화를 해야 하는데 이것은 pandas의 전문 영역이다.
객체 리스트로 받으면 규칙마다 수동 루프를 다시 짜게 된다.
    현실 → DB(장기 기억) → DataFrame(작업 기억) → 판단(RuleResult)
이 변환 사슬이 이 시스템의 데이터 흐름 전부다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RuleResult:
    """
    규칙 하나의 판단 결과 — "판단 세계"의 데이터 구조.

    [설명 가능성이 자료구조에 새겨진 모습]
    reason(사람이 읽는 근거 문장)과 evidence_ids(근거 거래 목록)가
    필수 필드로 박혀 있다. 즉 이 시스템에서는 근거 없는 판단이
    타입 수준에서 존재할 수 없다.
    "위험도 87%"만 뱉는 블랙박스와의 차이가 바로 이 두 필드다.
    """
    rule_name: str                 # 어떤 규칙의 판단인가
    risk_type: str                 # 의심 유형 (예: 쪼개기 송금 의심)
    is_suspicious: bool            # 의심 여부
    risk_score: int                # 이 규칙이 부여한 점수 (0~해당 규칙의 최대치)
    reason: str                    # 사람이 읽는 근거 문장 ← 설명 가능성의 핵심
    evidence_ids: list = field(default_factory=list)  # 근거 거래 ID 목록


class BaseFraudRule(ABC):
    """
    모든 의심 규칙의 추상 베이스.

    하위 클래스가 반드시 정하는 것:
      rule_name  — 규칙의 이름 (판단 결과와 DB에 기록된다)
      risk_type  — 이 규칙이 탐지하는 의심 유형
      max_score  — 이 규칙이 부여할 수 있는 최대 점수
                   (규칙별 심각도의 차이를 점수 상한으로 표현한다.
                    레이어링은 쪼개기 송금보다 은닉 의도가 강하므로
                    더 높은 상한을 갖는 식이다)
      evaluate() — 실제 판단 로직
    """

    rule_name: str = "base"
    risk_type: str = "미정의"
    max_score: int = 0

    @abstractmethod
    def evaluate(self, transactions_df: pd.DataFrame, **context) -> RuleResult:
        """
        거래 DataFrame을 검사하여 판단을 돌려준다.

        Parameters
        ----------
        transactions_df : 한 파산자(또는 한 계좌군)의 거래 테이블
        **context : 규칙이 추가로 필요로 하는 문맥.
                    예: filing_date(파산 신청일).
                    딕셔너리로 받는 이유 — 규칙마다 필요한 문맥이 다른데
                    시그니처를 고정하면 새 규칙이 낄 자리가 없어진다.

        Returns
        -------
        RuleResult : 근거가 포함된 판단
        """
        raise NotImplementedError

    def _clean_result(self) -> RuleResult:
        """의심 정황이 없을 때의 표준 응답. 하위 클래스 공용."""
        return RuleResult(
            rule_name=self.rule_name,
            risk_type=self.risk_type,
            is_suspicious=False,
            risk_score=0,
            reason="해당 유형의 의심 정황이 발견되지 않았다.",
        )
