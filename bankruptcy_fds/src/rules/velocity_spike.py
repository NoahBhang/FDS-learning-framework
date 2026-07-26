"""
bankruptcy-fds/src/rules/velocity_spike.py
============================================================
VelocitySpikeRule — 거래 급증(velocity spike) 탐지

[현실 문장]
"파산 신청 전 90일 동안, 특정 한 주(週)의 송금 건수나 총액이
평소(그 기간 내 주간 평균)보다 몇 배 이상 튀어 오르면, 자산을
서둘러 빼돌리려는 급조된 활동으로 의심한다."

[SplitTransferRule과의 교육적 차이]
쪼개기 송금은 "동일 수취인"이라는 동치류로 거래를 묶어 절대
횟수·금액을 본다. 이 규칙은 "같은 주"라는 동치류로 묶되,
절대 임계값이 아니라 그 debtor 자신의 다른 주들과 비교한
상대적 급증(spike)을 본다 — 정상적으로 거래가 잦은 사람과
평소 조용하다가 갑자기 활동이 튄 사람을 구분하기 위함이다.

[번역표: 현실 → 이산수학 → 코드]
  현실 개념              이산수학                   코드
  ---------------------  -------------------------  --------------------------------
  같은 주(週)             동치류로 분할               pd.Grouper(freq="W")
  평소보다 N배 이상        비율 비교 (avg 대비 배수)    count/total >= avg * multiplier
  "급증"이라 부를 최소한   바닥값(floor) 방어           min_week_count / min_week_amount
  "의심한다"              두 신호(건수/금액)의 논리합   OR

[알고리즘 복잡도]
groupby(freq="W")는 정렬 + 버킷화로 O(n log n) — 거래 규모가
커져도 실무 사용에 지장이 없다.
"""

import pandas as pd

from shared.rules.base_fraud_rule import BaseFraudRule, RuleResult


class VelocitySpikeRule(BaseFraudRule):
    rule_name = "velocity_spike"
    risk_type = "거래 급증(velocity spike) 의심"
    max_score = 25

    def __init__(self, window_days: int = 90,
                 spike_multiplier: float = 3.0,
                 min_week_count: int = 5,
                 min_week_amount: int = 3_000_000):
        """
        기준값을 생성자 파라미터로 받는 이유:
        "몇 배가 급증인가"는 조사 실무의 경험값이지 법 조문이 아니다.
        → "정책은 데이터, 로직은 코드"의 분리(SplitTransferRule과 동일 원칙).

        spike_multiplier : 주간 평균 대비 몇 배 이상이면 급증으로 보는가.
        min_week_count/min_week_amount : 평균 자체가 작을 때(예: 평소
            거래가 거의 없던 사람) multiplier만으로는 사소한 요동도
            "급증"으로 잡힐 수 있어, 최소 바닥값을 함께 요구한다.
        """
        self.window_days = window_days
        self.spike_multiplier = spike_multiplier
        self.min_week_count = min_week_count
        self.min_week_amount = min_week_amount

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

        # ── 1단계: 주(週) 단위 그룹화 ────────────────────────────
        # freq="W"는 기본적으로 일요일을 라벨로 하는 [월~일] 버킷이다.
        weekly = df.groupby(pd.Grouper(key="transaction_date", freq="W")).agg(
            count=("transaction_id", "size"),
            total=("amount", "sum"),
        )
        weekly = weekly[weekly["count"] > 0]

        # 비교할 다른 주가 없으면(활동 기간이 1주 이내) "급증"이라는
        # 상대 개념 자체를 판단할 기준선이 없다.
        if len(weekly) < 2:
            return self._clean_result()

        avg_count = weekly["count"].mean()
        avg_total = weekly["total"].mean()

        # ── 2단계: 급증 판정 (바닥값 AND 배수초과, 건수/금액 OR) ───
        spikes = weekly[
            ((weekly["count"] >= self.min_week_count)
             & (weekly["count"] >= avg_count * self.spike_multiplier))
            | ((weekly["total"] >= self.min_week_amount)
               & (weekly["total"] >= avg_total * self.spike_multiplier))
        ]
        if spikes.empty:
            return self._clean_result()

        # ── 3단계: 설명 생성 ────────────────────────────────────
        worst = spikes.sort_values("total", ascending=False).iloc[0]
        week_end = spikes.sort_values("total", ascending=False).index[0]
        week_start = week_end - pd.Timedelta(days=6)
        evidence = df[
            (df["transaction_date"] >= week_start)
            & (df["transaction_date"] <= week_end)
        ]["transaction_id"].tolist()

        count_multiple = worst["count"] / avg_count
        amount_multiple = worst["total"] / avg_total if avg_total > 0 else 0

        reason = (
            f"{week_start.date()}~{week_end.date()} 한 주 동안 "
            f"{int(worst['count'])}건, 총 {int(worst['total']):,}원이 송금되어 "
            f"해당 {self.window_days}일 기간 주간 평균(건수 {avg_count:.1f}건, "
            f"금액 {int(avg_total):,}원) 대비 각각 {count_multiple:.1f}배, "
            f"{amount_multiple:.1f}배 급증했다. "
            f"자산을 서둘러 처분하려는 급조된 활동 가능성이 있다."
        )

        # 점수 산정: 평균 대비 배수가 클수록 높은 점수, max_score로 상한.
        peak_multiple = max(count_multiple, amount_multiple)
        score = min(self.max_score, int(10 + 5 * peak_multiple))

        return RuleResult(
            rule_name=self.rule_name,
            risk_type=self.risk_type,
            is_suspicious=True,
            risk_score=score,
            reason=reason,
            evidence_ids=evidence,
        )
