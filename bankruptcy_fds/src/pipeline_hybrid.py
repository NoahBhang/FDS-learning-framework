"""
bankruptcy-fds/src/pipeline_hybrid.py
Phase 4+5: 룰 점수와 ML 점수의 융합, 위험도 레벨 부여,
SHAP 기반 의심 사유 자동 생성 → Evidence Packet 초안 출력.

융합 공식:  S_final = alpha * S_rule + (1 - alpha) * S_ml
  alpha는 룰에 대한 신뢰 비중. 초기 배포는 0.7(룰 우위)로 시작하고,
  ML 검증 성능이 쌓일수록 낮춘다 — "저항 없는 전환"의 운영 장치.

실행 예:
    python make_synth_case.py
    python pipeline_hybrid.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from shared.features import build_feature_matrix          # noqa: E402
from detect_anomaly import load_case, normalize_scores, FEATURES  # noqa: E402
from rules_mvp import RuleEngineMVP, REASONS              # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ALPHA = 0.7   # 룰 신뢰 비중 (배포 초기값)

# 위험도 레벨 (베타 설계 v1.0 기준)
LEVEL_BINS = [0.0, 0.40, 0.70, 0.90, 1.01]
LEVEL_NAMES = ["Low", "Medium", "High", "Critical"]

# SHAP 피처 → 관재인용 한국어 사유
FEATURE_REASONS = {
    "log_amount": "이례적인 금액 규모",
    "hour": "이례적인 거래 시간대",
    "is_night": "심야 시간대 거래",
    "is_weekend": "주말 거래",
    "is_related_party": "관계자 대상 거래",
    "memo_blank": "거래 메모 공란",
    "recipient_tx_count_30d": "동일 수취인 반복 송금",
    "same_day_recipient_count": "같은 날 반복 이체",
    "same_day_recipient_total": "같은 날 누적 고액 이체",
    "amount_ratio_vs_3m_avg": "일 거래량 급증",
    "days_to_filing": "파산 신청 임박 시점 거래",
}


def ml_reasons(shap_row: np.ndarray, top: int = 2) -> str:
    """가장 강하게 '이상' 방향으로 민 피처 top개를 사유 문구로 변환.

    IsolationForest의 SHAP 값은 낮을수록 이상 방향이므로,
    가장 음수인 기여부터 고른다.
    """
    order = np.argsort(shap_row)[:top]
    picked = [FEATURE_REASONS[FEATURES[i]] for i in order
              if shap_row[i] < 0]
    return ", ".join(picked) if picked else "-"


def main():
    print("1) 데이터 로드 & 피처 생성 (Phase 0~1)")
    tx, filing_date = load_case()
    feat = build_feature_matrix(tx, filing_date=filing_date)
    X = feat[FEATURES].fillna({"amount_ratio_vs_3m_avg": 1.0})
    y = feat["is_suspicious"]

    print("2) 룰 점수 (기존 설계의 학습판 아님 — 설계값 그대로)")
    rules = RuleEngineMVP().detect(feat)
    feat["rule_score"] = rules["rule_score"]
    feat["hit_patterns"] = rules["hit_patterns"]

    print("3) ML 점수 (Phase 3의 IsolationForest)")
    iso = IsolationForest(n_estimators=200,
                          max_samples=min(256, len(X)),
                          contamination="auto", random_state=42)
    iso.fit(X)
    feat["ml_score"] = normalize_scores(iso.score_samples(X))

    print("\n4) Phase 4 — 융합과 alpha 스윕 (라벨은 채점에만 사용)")
    print(f"   {'alpha':>6} | {'PR-AUC':>7} | 해석")
    for a in (1.0, 0.7, 0.5, 0.3, 0.0):
        fused = a * feat["rule_score"] + (1 - a) * feat["ml_score"]
        ap = average_precision_score(y, fused)
        tag = {1.0: "룰 단독", 0.0: "ML 단독", ALPHA: "배포 초기값"}.get(a, "")
        print(f"   {a:>6.1f} | {ap:>7.4f} | {tag}")

    feat["final_score"] = (ALPHA * feat["rule_score"]
                           + (1 - ALPHA) * feat["ml_score"])
    feat["risk_level"] = pd.cut(feat["final_score"], bins=LEVEL_BINS,
                                labels=LEVEL_NAMES, right=False)
    print("\n   위험도 레벨 분포:")
    print(feat["risk_level"].value_counts().reindex(LEVEL_NAMES).to_string())

    print("\n5) Phase 5 — SHAP 의심 사유 자동 생성")
    explainer = shap.TreeExplainer(iso)
    shap_values = explainer.shap_values(X)

    def build_reason(row_idx):
        rule_part = [REASONS[c] for c in feat.loc[row_idx, "hit_patterns"]]
        ml_part = ml_reasons(shap_values[feat.index.get_loc(row_idx)])
        return "; ".join(rule_part) + f" [ML 근거: {ml_part}]"

    top = feat.nlargest(10, "final_score").copy()
    top["의심사유"] = [build_reason(i) for i in top.index]

    print("\n   Evidence Packet 초안 (상위 10건):")
    cols = ["timestamp", "target_account", "amount",
            "final_score", "risk_level", "의심사유"]
    with pd.option_context("display.max_colwidth", 60):
        print(top[cols].to_string(
            index=False,
            formatters={"amount": "{:,.0f}".format,
                        "final_score": "{:.3f}".format}))

    out = DATA_DIR / "evidence_packet.csv"
    feat["의심사유"] = [
        build_reason(i) if feat.loc[i, "final_score"] >= 0.40 else ""
        for i in feat.index
    ]
    (feat.sort_values("final_score", ascending=False)
         .to_csv(out, index=False))
    print(f"\n저장 완료 → {out}")


if __name__ == "__main__":
    main()
