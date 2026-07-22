"""
bankruptcy-fds/src/rules_mvp.py
베타 설계 v1.0의 패턴 정의(P01~P04)를 피처 행렬 위에서 구현한 MVP 룰 엔진.
기본 점수·가중치·조합 보너스는 기존 설계 문서의 값을 그대로 따른다.

detect(feat) -> rule_score(0~1), hit_patterns(list) — shared/features.py의
add_rule_features()가 기대하는 Layer C 인터페이스와 동일하다.
"""

import numpy as np
import pandas as pd

REASONS = {
    "P01": "동일 수취인에게 단기간 반복 송금이 확인되었습니다",
    "P02": "고액 자금이 단기간에 분할 송금된 정황이 있습니다",
    "P03": "파산 신청 직전 평소 대비 거래 규모가 급증했습니다",
    "P04": "밀접 관계자로 분류된 대상에게 지급이 이전되었습니다",
}


class RuleEngineMVP:
    """피처 행렬(build_feature_matrix 출력)을 입력으로 받는 룰 엔진."""

    def detect(self, feat: pd.DataFrame) -> pd.DataFrame:
        f = feat
        ratio = f["amount_ratio_vs_3m_avg"].fillna(1.0).to_numpy()
        cnt30 = f["recipient_tx_count_30d"].to_numpy()

        # P01 반복 송금: 30일 내 3회 이상 (5회 +0.10, 10회 +0.20)
        p01 = cnt30 >= 3
        s01 = np.where(p01, 0.70 + 0.10 * (cnt30 >= 5) + 0.10 * (cnt30 >= 10), 0.0)

        # P02 분할 송금: 같은 날 2회 이상, 총액 500만원 이상
        p02 = ((f["same_day_recipient_count"] >= 2)
               & (f["same_day_recipient_total"] >= 5e6)).to_numpy()
        s02 = np.where(p02, 0.82, 0.0)

        # P03 신청 직전 급증: 신청 전 30일 & 일 거래량이 평균의 3배 초과
        p03 = (ratio > 3) & (f["days_to_filing"] <= 30).to_numpy()
        s03 = np.where(p03, 0.88 + 0.05 * (ratio > 5) + 0.05 * (ratio > 10), 0.0)

        # P04 친족/지인 송금: 관계자 목록 포함 (1천만원 이상 +0.10)
        p04 = (f["is_related_party"] == 1).to_numpy()
        s04 = np.where(p04, 0.80 + 0.10 * (f["amount"] >= 1e7).to_numpy(), 0.0)

        # 점수화: 히트한 패턴 중 최고 기본점수 + 조합형 보너스, 상한 1.0
        base = np.max(np.vstack([s01, s02, s03, s04]), axis=0)
        bonus = 0.15 * (p01 & p04) + 0.18 * (p02 & p03)
        rule_score = np.minimum(1.0, base + bonus)

        hits = [
            [c for c, flag in zip(("P01", "P02", "P03", "P04"),
                                  (a, b, cc, d)) if flag]
            for a, b, cc, d in zip(p01, p02, p03, p04)
        ]
        return pd.DataFrame(
            {"rule_score": rule_score, "hit_patterns": hits}, index=f.index
        )
