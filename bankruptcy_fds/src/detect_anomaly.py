"""
bankruptcy-fds/src/detect_anomaly.py
Phase 3: IsolationForest 비지도 이상탐지.

라벨(is_suspicious)은 학습에 절대 쓰지 않는다 — 실제 파산자 데이터에는
라벨이 없기 때문이다. 라벨은 마지막 채점(PR-AUC) 단계에서만 사용해
"라벨 없이 얼마나 잡았는가"를 측정한다.

실행 예 (fds-project conda 환경):
    python make_synth_case.py
    python detect_anomaly.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

# 저장소 루트를 경로에 추가해 shared 모듈을 재사용
sys.path.append(str(Path(__file__).resolve().parents[2]))
from shared.features import build_feature_matrix  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# 파산 FDS용 피처: PaySim과 달리 단일 파산자 데이터이므로
# amount_ratio_vs_3m_avg(일 거래량 급증)와 days_to_filing이 의미를 가진다.
FEATURES = [
    "log_amount", "hour", "is_night", "is_weekend",
    "is_related_party", "memo_blank",
    "recipient_tx_count_30d",
    "same_day_recipient_count", "same_day_recipient_total",
    "amount_ratio_vs_3m_avg", "days_to_filing",
]
LABEL = "is_suspicious"          # 채점 전용. FEATURES에 절대 넣지 않는다.


def load_case():
    tx = pd.read_csv(DATA_DIR / "transactions.csv",
                     parse_dates=["timestamp"])
    related = pd.read_csv(DATA_DIR / "related_parties.csv")
    meta = pd.read_csv(DATA_DIR / "case_meta.csv",
                       parse_dates=["filing_date"])
    tx["is_related_party"] = (
        tx["target_account"].isin(related["target_account"]).astype(int)
    )
    return tx, meta["filing_date"].iloc[0]


def normalize_scores(raw: np.ndarray) -> np.ndarray:
    """sklearn score_samples는 음수이며 낮을수록 이상하다.
    부호를 뒤집고 0~1로 정규화해 '높을수록 위험'으로 통일한다
    (룰 점수와 같은 방향 — Phase 4 융합을 위한 정지 작업)."""
    flipped = -raw
    lo, hi = flipped.min(), flipped.max()
    return (flipped - lo) / (hi - lo) if hi > lo else np.zeros_like(flipped)


def precision_at_k(y_true, scores, k):
    idx = np.argsort(np.asarray(scores))[::-1][:k]
    return np.asarray(y_true)[idx].mean()


def main():
    print("1) 사건 데이터 로드 (3종 세트)")
    tx, filing_date = load_case()
    print(f"   거래 {len(tx):,}건 | 신청일 {filing_date.date()}")

    print("2) 피처 생성 (shared/features.py — PaySim과 동일 파이프라인)")
    feat = build_feature_matrix(tx, filing_date=filing_date)
    X = feat[FEATURES].fillna({"amount_ratio_vs_3m_avg": 1.0})

    print("3) IsolationForest 학습 (라벨 미사용)")
    iso = IsolationForest(
        n_estimators=200,
        max_samples=min(256, len(X)),   # 서브샘플링: swamping/masking 완화
        contamination="auto",
        random_state=42,
    )
    iso.fit(X)
    feat["ml_score"] = normalize_scores(iso.score_samples(X))

    print("4) 라벨로 채점 (여기서만 정답 사용)")
    y = feat[LABEL]
    base = y.mean()
    ap = average_precision_score(y, feat["ml_score"])
    print(f"   PR-AUC: {ap:.4f} | 무작위 기준선 {base:.4f} 대비 {ap / base:.1f}배")
    n_pos = int(y.sum())
    for k in (10, n_pos, 50):
        p = precision_at_k(y, feat["ml_score"], k)
        rec = p * k / n_pos
        print(f"   상위 {k}건 조사 시: 정밀도 {p:.3f}"
              f" | 전체 의심거래({n_pos}건)의 {rec * 100:.0f}% 회수")

    # 패턴별 회수율: 어떤 수법을 잘/못 잡는지 진단
    top_n = feat.nlargest(n_pos, "ml_score")
    print("\n   패턴별 상위권 진입 (심은 건수 대비):")
    planted = feat[feat[LABEL] == 1]["true_pattern"].value_counts()
    caught = top_n[top_n[LABEL] == 1]["true_pattern"].value_counts()
    for pat in planted.index:
        print(f"     {pat}: {caught.get(pat, 0)}/{planted[pat]}건")

    print("\n5) 관재인용 상위 의심 거래 (Evidence Packet 후보)")
    cols = ["timestamp", "target_account", "amount",
            "is_related_party", "ml_score", "true_pattern"]
    print(feat.nlargest(10, "ml_score")[cols]
          .to_string(index=False,
                     formatters={"amount": "{:,.0f}".format,
                                 "ml_score": "{:.3f}".format}))

    out = DATA_DIR / "scored_transactions.csv"
    feat.sort_values("ml_score", ascending=False).to_csv(out, index=False)
    print(f"\n저장 완료 → {out}")


if __name__ == "__main__":
    main()
