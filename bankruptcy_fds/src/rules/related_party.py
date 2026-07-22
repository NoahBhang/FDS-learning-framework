"""
bankruptcy-fds/src/rules/related_party.py
============================================================
RelatedPartyRule — 편파 변제(특수관계인 집중 송금) 탐지

[현실 문장]
"파산 신청 전 90일 동안의 송금 총액 중 가족·지인에게 간 비율이
50%를 넘으면, 특정 채권자에게만 변제한 편파 변제를 의심한다."

[SplitTransferRule과의 교육적 차이]
쪼개기 송금은 절대량(횟수, 총액)의 규칙이고,
편파 변제는 비율의 규칙이다.

  비율 = (특수관계인에게 간 금액) / (전체 송금액)

같은 500만 원이라도 전 재산이 600만 원인 사람과 6억인 사람은
의미가 다르다. 절대량 규칙과 비율 규칙을 함께 두어야
FDS가 규모가 다른 사건들을 공정하게 본다.
"""

import pandas as pd

from shared.rules.base_fraud_rule import BaseFraudRule, RuleResult


class RelatedPartyRule(BaseFraudRule):
    rule_name = "related_party"
    risk_type = "편파 변제 의심"
    max_score = 30

    # 특수관계인으로 간주하는 관계 유형. 집합(set)으로 두는 이유:
    # "관계가 이 집합에 속하는가"(isin)는 소속 판정이고,
    # 소속 판정의 자연스러운 자료구조가 집합이다.
    RELATED_TYPES = {"가족", "지인"}

    def __init__(self, window_days: int = 90, ratio_threshold: float = 0.5):
        self.window_days = window_days
        self.ratio_threshold = ratio_threshold

    def evaluate(self, transactions_df: pd.DataFrame, **context) -> RuleResult:
        filing_date = context["filing_date"]

        window_start = filing_date - pd.Timedelta(days=self.window_days)
        df = transactions_df[
            (transactions_df["transaction_type"] == "송금")
            & (transactions_df["transaction_date"] >= window_start)
            & (transactions_df["transaction_date"] <= filing_date)
        ]
        if df.empty:
            return self._clean_result()

        total_amount = df["amount"].sum()
        related = df[df["relation_type"].isin(self.RELATED_TYPES)]
        related_amount = related["amount"].sum()

        # 0으로 나누기 방어 — 현실 데이터의 불완전성 대응(1부 9장 예외 처리 관점).
        if total_amount == 0:
            return self._clean_result()

        ratio = related_amount / total_amount
        if ratio < self.ratio_threshold:
            return self._clean_result()

        reason = (
            f"파산 신청 전 {self.window_days}일 동안의 송금 총액 "
            f"{int(total_amount):,}원 중 {int(related_amount):,}원"
            f"({ratio:.0%})이 가족·지인에게 집중되었다. "
            f"일반 채권자를 배제한 편파 변제 가능성이 있다."
        )

        # 비율이 기준을 초과한 정도에 비례한 점수.
        score = min(self.max_score, int(self.max_score * ratio))

        return RuleResult(
            rule_name=self.rule_name,
            risk_type=self.risk_type,
            is_suspicious=True,
            risk_score=score,
            reason=reason,
            evidence_ids=related["transaction_id"].tolist(),
        )
