"""
kaggle-bank-fds/src/train.py
Phase 2: Logistic Regression 베이스라인 → LightGBM, PR-AUC 평가.

실행 예 (fds-project conda 환경):
    python train.py ../data/PS_20174392719_1491204439457_log.csv --max-step 300
    python train.py ../data/PS_20174392719_1491204439457_log.csv   # 전체 실행
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 저장소 루트를 경로에 추가해 shared 모듈을 재사용
sys.path.append(str(Path(__file__).resolve().parents[2]))
from shared.features import add_raw_features, add_aggregate_features  # noqa: E402
from paysim_adapter import load_paysim, to_action_log                 # noqa: E402

FEATURES = [
    # Layer A: 원시 피처 (shared/features.py)
    "log_amount", "hour", "is_night", "is_weekend",
    # Layer B: 집계 피처 (shared/features.py) — 룰의 연속형 승격판
    "recipient_tx_count_30d",
    "same_day_recipient_count", "same_day_recipient_total",
    # PaySim 고유 신호 (paysim_adapter.py)
    "balance_error_orig", "balance_error_dest",
    "orig_emptied", "is_transfer",
]
TARGET = "isFraud"


def time_split(df, train_frac=0.70, valid_frac=0.15):
    """무작위 분할 대신 시간 기반 분할.

    FDS는 과거로 학습해 미래를 판정하는 시스템이므로,
    무작위 분할은 미래 정보가 학습에 섞이는 누수를 일으킨다.
    train(앞 70%) / valid(다음 15%, 조기 종료용) / test(마지막 15%).
    """
    cut_tr = df["step"].quantile(train_frac)
    cut_va = df["step"].quantile(train_frac + valid_frac)
    train = df[df["step"] <= cut_tr]
    valid = df[(df["step"] > cut_tr) & (df["step"] <= cut_va)]
    test = df[df["step"] > cut_va]
    return train, valid, test


def precision_at_k(y_true, scores, k):
    """상위 k건을 조사 대상으로 올렸을 때의 정밀도 — 관재인 관점의 지표."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    idx = np.argsort(scores)[::-1][:k]
    return y_true[idx].mean()


def report(name, y_true, scores, base_rate):
    y_arr = np.asarray(y_true)
    n_pos = int(y_arr.sum())
    ap = average_precision_score(y_true, scores)
    lift = ap / base_rate if base_rate > 0 else float("nan")
    print(f"\n[{name}]  (테스트 구간 실제 사기: {n_pos:,}건)")
    print(f"  PR-AUC (average precision): {ap:.4f}"
          f"  | 무작위 기준선 {base_rate:.4f} 대비 {lift:,.0f}배")
    for k in (100, 500):
        k_eff = min(k, len(scores))
        p = precision_at_k(y_arr, scores, k_eff)
        tp_topk = int(round(p * k_eff))
        rec = tp_topk / n_pos if n_pos > 0 else float("nan")
        print(f"  상위 {k_eff}건 조사 시: 정밀도 {p:.3f}"
              f" | 전체 사기의 {rec * 100:.1f}% 회수")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="PaySim CSV 경로")
    parser.add_argument("--max-step", type=int, default=None,
                        help="첫 실행 검증용: 처음 N시간 분량만 사용")
    args = parser.parse_args()

    print("1) 데이터 로드 & 어댑터 변환")
    raw = load_paysim(args.csv_path, max_step=args.max_step)
    log = to_action_log(raw)
    print(f"   {len(log):,}행 | 사기 비율 {log[TARGET].mean():.4%}")

    print("2) 피처 생성 (shared/features.py 재사용)")
    feat = add_raw_features(log)
    feat = add_aggregate_features(feat)

    print("3) 시간 기반 분할 (train 70% / valid 15% / test 15%)")
    train, valid, test = time_split(feat)
    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_va, y_va = valid[FEATURES], valid[TARGET]
    X_te, y_te = test[FEATURES], test[TARGET]
    base_rate = y_te.mean()
    print(f"   test 사기 비율(= PR-AUC 무작위 기준선): {base_rate:.4f}")

    print("\n4) 베이스라인: Logistic Regression")
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    logreg.fit(X_tr, y_tr)
    report("Logistic Regression", y_te, logreg.predict_proba(X_te)[:, 1],
           base_rate)

    # 계수 = "데이터가 학습한 룰 가중치"
    coefs = pd.Series(logreg[-1].coef_[0], index=FEATURES)
    print("\n  학습된 계수 (표준화 기준, 절댓값 큰 순):")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index)
          .round(3).to_string())

    print("\n5) LightGBM")
    pos = max(int(y_tr.sum()), 1)
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        scale_pos_weight=(len(y_tr) - pos) / pos,  # 클래스 불균형 보정
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    report("LightGBM", y_te, model.predict_proba(X_te)[:, 1], base_rate)
    print(f"\n  조기 종료 시점: {model.best_iteration_}번째 트리")

    imp = pd.Series(model.feature_importances_, index=FEATURES)
    print("\n  피처 중요도 (분기 사용 횟수 기준):")
    print(imp.sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
