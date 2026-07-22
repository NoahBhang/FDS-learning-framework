"""
shared/features.py
룰 기반 FDS의 5가지 패턴을 연속형 ML 피처로 승격시키는 모듈.
파산관재인 FDS와 Kaggle PaySim FDS가 공유한다.

[v1.1 수정] recipient_tx_count_30d 계산 시 groupby.rolling 결과와
원본 행 순서가 어긋나는 정렬 버그 수정:
(target_account, timestamp)로 먼저 정렬한 뒤 rolling(on="timestamp")을
사용해 행 순서를 일치시킨다.
"""

import numpy as np
import pandas as pd


def add_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Layer A: 원시 피처 — 거래 한 건 자체의 속성"""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])

    # 금액 분포는 극도로 치우쳐 있으므로 로그 변환
    out["log_amount"] = np.log1p(out["amount"])

    # 시간 피처
    out["hour"] = out["timestamp"].dt.hour
    out["is_night"] = out["hour"].between(0, 5).astype(int)
    out["is_weekend"] = (out["timestamp"].dt.dayofweek >= 5).astype(int)

    # P04: 관계자 여부 (관계자 목록 병합이 선행되었다고 가정)
    if "is_related_party" not in out.columns:
        out["is_related_party"] = 0

    # P05: 명의 불일치
    if "name_mismatch" not in out.columns:
        out["name_mismatch"] = 0

    # 메모 공란 여부 (메모 개념이 없는 데이터셋은 1로 고정)
    if "memo" in out.columns:
        out["memo_blank"] = out["memo"].isna().astype(int)
    else:
        out["memo_blank"] = 1

    return out


def add_aggregate_features(df: pd.DataFrame,
                           filing_date: pd.Timestamp | None = None
                           ) -> pd.DataFrame:
    """Layer B: 집계 피처 — 룰의 임계값 조건을 연속형으로 계산"""
    # rolling 결과와 행 순서를 일치시키기 위해 (수취인, 시간)으로 정렬
    out = df.sort_values(["target_account", "timestamp"]).copy()

    # --- P01 승격: 동일 수취인 30일 이동 송금 횟수 (과거만 참조) ---
    out["recipient_tx_count_30d"] = (
        out.groupby("target_account")
           .rolling("30D", on="timestamp")["amount"]
           .count()
           .values
    )

    # --- P02 승격: 같은 날 · 동일 수취인 분할 신호 ---
    out["date"] = out["timestamp"].dt.date
    grp = out.groupby(["target_account", "date"])["amount"]
    out["same_day_recipient_count"] = grp.transform("count")
    out["same_day_recipient_total"] = grp.transform("sum")

    # --- P03 승격: 최근 3개월 일평균 대비 배율 ---
    # 주의: 단일 파산자 데이터에서 의미가 있는 피처.
    # PaySim처럼 수백만 계좌가 섞인 데이터에서는 사용하지 않는다.
    daily = out.groupby("date")["amount"].sum()
    rolling_avg_90d = daily.rolling(90, min_periods=7).mean()
    ratio = (daily / rolling_avg_90d).rename("amount_ratio_vs_3m_avg")
    out = out.merge(ratio, left_on="date", right_index=True, how="left")

    if filing_date is not None:
        out["days_to_filing"] = (
            (filing_date - out["timestamp"]).dt.days.clip(lower=0)
        )

    return out.sort_values("timestamp").drop(columns=["date"])


def add_rule_features(df: pd.DataFrame, rule_engine) -> pd.DataFrame:
    """
    Layer C: 룰 출력 피처 — 기존 룰 엔진의 판정을 모델 입력으로.
    rule_engine.detect(df)가 거래별 rule_score, hit_patterns를
    반환한다고 가정 (기존 fds_engine 인터페이스 준용).
    """
    out = df.copy()
    results = rule_engine.detect(out)
    out["rule_score"] = results["rule_score"]
    out["rule_hit_count"] = results["hit_patterns"].str.len()
    return out


def build_feature_matrix(df: pd.DataFrame,
                         rule_engine=None,
                         filing_date=None) -> pd.DataFrame:
    """세 계층을 순서대로 쌓아 최종 피처 행렬을 만든다."""
    out = add_raw_features(df)
    out = add_aggregate_features(out, filing_date=filing_date)
    if rule_engine is not None:
        out = add_rule_features(out, rule_engine)
    return out
