"""
bankruptcy-fds/src/rules/split_transfer.py
============================================================
SplitTransferRule — 쪼개기 송금 탐지

[현실 문장]
"파산자가 파산 신청 전 30일 동안 동일 수취인에게 5회 이상
송금했고, 총액이 500만 원을 넘으면 쪼개기 송금으로 의심한다."

[번역표: 현실 → 이산수학 → 코드]  (2부 확장편 22장의 실물)

  현실 개념            이산수학                코드
  ------------------  ---------------------  ---------------------------
  신청 전 30일 이내    구간 조건 (부등식)       date >= filing - 30일
  동일 수취인          동치류로 분할            groupby("receiver")
  5회 이상            집합의 크기 |S| >= 5     count >= min_count
  총액 500만 원 이상   합 Σ >= threshold       sum >= threshold
  "의심한다"          네 명제의 논리곱(AND)    모든 조건 동시 충족

[알고리즘 복잡도]
전체 거래를 한 번 필터링하고(O(n)) 한 번 그룹화한다(O(n)).
따라서 전체 O(n) — 거래 100만 건에도 실무 사용이 가능하다.
"""

import pandas as pd

from shared.rules.base_fraud_rule import BaseFraudRule, RuleResult


class SplitTransferRule(BaseFraudRule):
    rule_name = "split_transfer"
    risk_type = "쪼개기 송금 의심"
    max_score = 35

    def __init__(self, window_days: int = 30,
                 min_count: int = 5,
                 total_amount_threshold: int = 5_000_000):
        """
        기준값을 생성자 파라미터로 받는 이유:
        30일·5회·500만 원은 법 조문이 아니라 조사 실무의 경험값이다.
        관재인마다, 사건마다 조정될 수 있으므로 코드에 박지 않고
        (하드코딩하지 않고) 바깥에서 주입받는다.
        → "정책은 데이터, 로직은 코드"의 분리.
        """
        self.window_days = window_days
        self.min_count = min_count
        self.total_amount_threshold = total_amount_threshold

    def evaluate(self, transactions_df: pd.DataFrame, **context) -> RuleResult:
        filing_date = context["filing_date"]   # 파산 신청일 (Debtor의 속성)

        # ── 1단계: 시간 창(window) 필터 ──────────────────────────
        # "신청 전 30일 이내의 송금"이라는 현실 조건을
        # 불리언 마스크 두 개의 AND로 번역한다.
        window_start = filing_date - pd.Timedelta(days=self.window_days)
        df = transactions_df[
            (transactions_df["transaction_type"] == "송금")
            & (transactions_df["transaction_date"] >= window_start)
            & (transactions_df["transaction_date"] <= filing_date)
        ]
        if df.empty:
            return self._clean_result()

        # ── 2단계: 수취인별 그룹화 ───────────────────────────────
        # groupby는 "수취인이 같다"는 동치관계로 거래 집합을 분할한다.
        # 딕셔너리 {수취인: 거래들}를 pandas가 대신 만들어 주는 것이다.
        grouped = df.groupby("receiver").agg(
            count=("transaction_id", "size"),
            total=("amount", "sum"),
        )

        # ── 3단계: 논리곱(AND) 판정 ─────────────────────────────
        hits = grouped[
            (grouped["count"] >= self.min_count)
            & (grouped["total"] >= self.total_amount_threshold)
        ]
        if hits.empty:
            return self._clean_result()

        # ── 4단계: 설명 생성 ────────────────────────────────────
        # 판단은 반드시 근거 문장과 근거 거래 목록을 동반한다.
        # 가장 심한 수취인 1명을 대표 사례로 문장화한다.
        worst = hits.sort_values("total", ascending=False).iloc[0]
        worst_receiver = hits.sort_values("total", ascending=False).index[0]
        evidence = df[df["receiver"] == worst_receiver]["transaction_id"].tolist()

        reason = (
            f"파산 신청 전 {self.window_days}일 동안 동일 수취인 "
            f"'{worst_receiver}'에게 {int(worst['count'])}회 송금되었고, "
            f"총액이 {int(worst['total']):,}원이다. "
            f"개별 금액을 분산하여 합산 금액을 은폐하려는 "
            f"쪼개기 송금 가능성이 있다."
        )

        # 점수 산정: 기준 초과 정도에 비례하되 max_score로 상한.
        over_ratio = worst["total"] / self.total_amount_threshold
        score = min(self.max_score, int(20 + 5 * over_ratio))

        return RuleResult(
            rule_name=self.rule_name,
            risk_type=self.risk_type,
            is_suspicious=True,
            risk_score=score,
            reason=reason,
            evidence_ids=evidence,
        )
