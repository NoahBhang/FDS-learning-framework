"""
bankruptcy-fds/src/rules/temporal_anomaly.py
============================================================
TemporalAnomalyRule — 비정상 시간대 활동 탐지

[현실 문장]
"파산 신청 전 90일 동안의 송금이 야간(00~06시) 또는 주말에
비정상적으로 집중되어 있으면, 정상적인 상거래 활동이 아니라
남의 눈을 피하려는 은닉 시도로 의심한다."

[스키마 제약과 이 규칙이 그것에 대응하는 방식]
database/schema.sql의 transactions.transaction_date는 DATE 타입이다
(시각 없음). DataLoader.get_transactions()가 parse_dates로 읽으면
pandas는 이를 자정(00:00:00)으로 채운 Timestamp로 만든다.
즉 이 DB 스키마만 놓고 보면 모든 거래의 시(hour)가 항상 0이 되어
"야간 비율"이 아무 정보도 없이 100%로 계산되는 착시가 생긴다.

이 규칙은 그 착시를 막기 위해, 주어진 데이터의 hour 값이 실제로
다양한지(has_time_resolution) 먼저 확인한다. 시각 정보가 없다고
판단되면 야간 신호를 끄고 주말 신호만으로 판단한다 — 없는 데이터로
있는 척 판단하지 않는다는 원칙이다. (관계자 비율 규칙의 "0으로
나누기 방어"와 같은 종류의 현실 데이터 불완전성 대응이다.)
시각까지 기록하는 스키마로 확장되면(예: transaction_datetime 컬럼
추가) 이 가드는 자동으로 비활성화되고 야간 신호가 정상 작동한다.

[SplitTransferRule / RelatedPartyRule과의 교육적 차이]
쪼개기 송금·편파 변제는 "누구에게"의 문제였다. 이 규칙은 "언제"의
문제다 — 같은 논리곱/비율 사고를 시간 축에 적용한 것이다.
"""

import pandas as pd

from shared.rules.base_fraud_rule import BaseFraudRule, RuleResult


class TemporalAnomalyRule(BaseFraudRule):
    rule_name = "temporal_anomaly"
    risk_type = "비정상 시간대 거래 의심"
    max_score = 20

    NIGHT_HOURS = range(0, 6)  # 00:00 ~ 05:59

    def __init__(self, window_days: int = 90,
                 night_ratio_threshold: float = 0.5,
                 weekend_ratio_threshold: float = 0.5,
                 min_count: int = 5):
        """
        기준값을 생성자 파라미터로 받는 이유: 몇 %가 "집중"인가는
        조사 실무의 경험값이다 (SplitTransferRule과 동일 원칙).

        min_count : 표본이 너무 작으면 비율(ratio)이 우연히 튈 수 있다.
            예컨대 거래 2건이 모두 주말이면 비율은 100%이지만 근거로
            삼기엔 표본이 너무 작다 — 최소 관측 건수로 이를 방어한다.
        """
        self.window_days = window_days
        self.night_ratio_threshold = night_ratio_threshold
        self.weekend_ratio_threshold = weekend_ratio_threshold
        self.min_count = min_count

    def evaluate(self, transactions_df: pd.DataFrame, **context) -> RuleResult:
        filing_date = context["filing_date"]

        window_start = filing_date - pd.Timedelta(days=self.window_days)
        df = transactions_df[
            (transactions_df["transaction_type"] == "송금")
            & (transactions_df["transaction_date"] >= window_start)
            & (transactions_df["transaction_date"] <= filing_date)
        ]
        if len(df) < self.min_count:
            return self._clean_result()

        # ── 1단계: 주말 비율 ─────────────────────────────────────
        weekend_mask = df["transaction_date"].dt.dayofweek >= 5   # 토(5)·일(6)
        weekend_ratio = weekend_mask.mean()

        # ── 2단계: 야간 비율 (시각 정보가 실제로 있을 때만) ────────
        hours = df["transaction_date"].dt.hour
        has_time_resolution = hours.nunique() > 1
        night_mask = hours.isin(self.NIGHT_HOURS) if has_time_resolution \
            else pd.Series(False, index=df.index)
        night_ratio = night_mask.mean() if has_time_resolution else 0.0

        # ── 3단계: 논리합(OR) 판정 ───────────────────────────────
        is_night_hit = has_time_resolution and night_ratio >= self.night_ratio_threshold
        is_weekend_hit = weekend_ratio >= self.weekend_ratio_threshold
        if not (is_night_hit or is_weekend_hit):
            return self._clean_result()

        # ── 4단계: 설명 생성 ────────────────────────────────────
        hit_mask = pd.Series(False, index=df.index)
        parts = []
        if is_night_hit:
            hit_mask |= night_mask
            parts.append(f"야간(00~06시) 거래 비율 {night_ratio:.0%}")
        if is_weekend_hit:
            hit_mask |= weekend_mask
            parts.append(f"주말 거래 비율 {weekend_ratio:.0%}")

        reason = (
            f"파산 신청 전 {self.window_days}일 동안 총 {len(df)}건의 송금 중 "
            + " · ".join(parts) + "로 비정상 시간대에 활동이 집중되었다. "
            "정상적인 상거래 패턴에서 벗어난 은닉 시도 가능성이 있다."
        )

        # 점수 산정: 두 신호 중 더 강한 쪽의 비율에 비례, max_score로 상한.
        peak_ratio = max(
            night_ratio if is_night_hit else 0.0,
            weekend_ratio if is_weekend_hit else 0.0,
        )
        score = min(self.max_score, int(self.max_score * peak_ratio))

        return RuleResult(
            rule_name=self.rule_name,
            risk_type=self.risk_type,
            is_suspicious=True,
            risk_score=score,
            reason=reason,
            evidence_ids=df[hit_mask]["transaction_id"].tolist(),
        )
