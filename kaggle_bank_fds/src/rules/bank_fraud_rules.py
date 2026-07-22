"""
kaggle-bank-fds/src/rules/bank_fraud_rules.py
============================================================
은행권 FDS 규칙 — 같은 사고틀, 다른 도메인

[이 파일의 존재 이유가 곧 포트폴리오의 핵심 메시지]
파산관재인 FDS와 은행권 FDS는 도메인이 다르다.
그러나 두 시스템 모두 shared.rules.BaseFraudRule을 상속한다.
즉 "판단은 evaluate()로, 결과는 근거를 동반한 RuleResult로"라는
사고틀이 도메인을 넘어 재사용됨을 증명한다.
면접에서 "같은 프레임워크를 두 도메인에 적용했다"고 말할 때
가리킬 파일이 바로 이것이다.

[PaySim 데이터셋의 컬럼 (Kaggle)]
  step           : 시뮬레이션 시간 (1 step = 1시간)
  type           : 거래 유형 (TRANSFER, CASH_OUT, PAYMENT, ...)
  amount         : 금액
  nameOrig       : 보낸 계좌
  oldbalanceOrg  : 보낸 계좌의 거래 전 잔액
  newbalanceOrig : 보낸 계좌의 거래 후 잔액
  nameDest       : 받은 계좌
  isFraud        : 정답 레이블 (모델 평가용 — 규칙은 이 컬럼을 보지 않는다!)
"""

import pandas as pd

from shared.rules.base_fraud_rule import BaseFraudRule, RuleResult


class TransferCashOutRule(BaseFraudRule):
    """
    TRANSFER → CASH_OUT 연쇄 탐지.

    [현실 문장]
    "어떤 계좌로 이체(TRANSFER)된 직후, 그 계좌에서 비슷한 금액이
    현금 인출(CASH_OUT)되면 자금 세탁의 전형적 수법을 의심한다."

    파산 FDS의 레이어링과 본질이 같다:
    돈이 '거쳐 가는' 패턴의 탐지다. 도메인 언어만 다를 뿐
    "경로를 추적한다"는 사고는 동일하다.
    """
    rule_name = "transfer_cash_out"
    risk_type = "이체 후 즉시 현금화 의심"
    max_score = 40

    def __init__(self, max_step_gap: int = 24, amount_tolerance: float = 0.05):
        self.max_step_gap = max_step_gap          # 이체~인출 사이 최대 시간(스텝)
        self.amount_tolerance = amount_tolerance  # 금액 허용 오차 (수수료 감안 5%)

    def evaluate(self, transactions_df: pd.DataFrame, **context) -> RuleResult:
        transfers = transactions_df[transactions_df["type"] == "TRANSFER"]
        cashouts = transactions_df[transactions_df["type"] == "CASH_OUT"]
        if transfers.empty or cashouts.empty:
            return self._clean_result()

        # 이체의 목적지 계좌(nameDest)가 인출의 출발 계좌(nameOrig)인 쌍을 찾는다.
        # merge = 관계대수의 조인. "두 사건이 같은 계좌를 공유한다"는
        # 관계를 테이블 연산으로 표현한 것이다.
        merged = transfers.merge(
            cashouts,
            left_on="nameDest", right_on="nameOrig",
            suffixes=("_in", "_out"),
        )
        if merged.empty:
            return self._clean_result()

        # 시간 조건: 인출이 이체 이후, max_step_gap 이내
        # 금액 조건: 인출액이 이체액의 ±5% 이내 (수수료를 감안한 근사 일치)
        hits = merged[
            (merged["step_out"] >= merged["step_in"])
            & (merged["step_out"] - merged["step_in"] <= self.max_step_gap)
            & ((merged["amount_out"] - merged["amount_in"]).abs()
               <= merged["amount_in"] * self.amount_tolerance)
        ]
        if hits.empty:
            return self._clean_result()

        top = hits.sort_values("amount_in", ascending=False).iloc[0]
        reason = (
            f"계좌 {top['nameOrig_in']}에서 {top['nameDest_in']}로 "
            f"{top['amount_in']:,.0f}이 이체된 뒤 "
            f"{int(top['step_out'] - top['step_in'])}시간 이내에 "
            f"거의 동일한 금액({top['amount_out']:,.0f})이 현금 인출되었다. "
            f"총 {len(hits)}건의 이체-즉시-현금화 패턴이 발견되었다."
        )
        score = min(self.max_score, 20 + 5 * len(hits))

        return RuleResult(
            rule_name=self.rule_name,
            risk_type=self.risk_type,
            is_suspicious=True,
            risk_score=score,
            reason=reason,
            evidence_ids=hits.index.tolist(),
        )


class FullBalanceTransferRule(BaseFraudRule):
    """
    계좌 전액 이체 탐지.

    [현실 문장]
    "계좌 잔액 전부를 한 번에 이체하는 것은 계좌 탈취 후
    자금을 빼돌리는 전형적 신호다."

    파산 FDS의 편파 변제처럼 이것도 비율의 규칙이다:
      이체액 / 거래 전 잔액 = 1.0 (전액)
    도메인은 달라도 "절대량이 아니라 비율을 본다"는 사고가 재사용된다.
    """
    rule_name = "full_balance_transfer"
    risk_type = "계좌 전액 이체 의심"
    max_score = 30

    def evaluate(self, transactions_df: pd.DataFrame, **context) -> RuleResult:
        transfers = transactions_df[transactions_df["type"] == "TRANSFER"]
        if transfers.empty:
            return self._clean_result()

        # 잔액이 0이 아닌 계좌에서, 이체액 == 거래 전 잔액 (전액 이체)
        hits = transfers[
            (transfers["oldbalanceOrg"] > 0)
            & (transfers["amount"] >= transfers["oldbalanceOrg"] * 0.999)
        ]
        if hits.empty:
            return self._clean_result()

        total = hits["amount"].sum()
        reason = (
            f"계좌 잔액 전액을 한 번에 이체한 거래가 {len(hits)}건 "
            f"발견되었다 (합계 {total:,.0f}). "
            f"계좌 탈취 또는 자금 도피의 전형적 패턴이다."
        )
        score = min(self.max_score, 15 + 5 * len(hits))

        return RuleResult(
            rule_name=self.rule_name,
            risk_type=self.risk_type,
            is_suspicious=True,
            risk_score=score,
            reason=reason,
            evidence_ids=hits.index.tolist(),
        )
