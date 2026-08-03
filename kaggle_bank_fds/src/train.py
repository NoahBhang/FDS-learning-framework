"""Train and explicitly persist the versioned PaySim LightGBM artifact."""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kaggle_bank_fds.src.models.paysim_ml_contract import (
    PAYSIM_ARTIFACT_SCHEMA_VERSION,
    PAYSIM_DEFAULT_THRESHOLD,
    PAYSIM_FEATURE_NAMES,
    PAYSIM_FEATURE_SCHEMA_VERSION,
    PAYSIM_MODEL_NAME,
    validate_paysim_artifact_bundle,
)
from kaggle_bank_fds.src.models.predictor import (
    DEFAULT_PAYSIM_MODEL_PATH,
    PaySimFraudPredictor,
    prepare_paysim_features,
)

FEATURES = list(PAYSIM_FEATURE_NAMES)
TARGET = "isFraud"
DEFAULT_MODEL_VERSION = "paysim-lightgbm-v1"


def time_split(df: pd.DataFrame, train_frac: float = 0.70, valid_frac: float = 0.15):
    """Split chronologically into train, validation, and test sets."""
    cut_tr = df["step"].quantile(train_frac)
    cut_va = df["step"].quantile(train_frac + valid_frac)
    train = df[df["step"] <= cut_tr]
    valid = df[(df["step"] > cut_tr) & (df["step"] <= cut_va)]
    test = df[df["step"] > cut_va]
    return train, valid, test


def precision_at_k(y_true, scores, k):
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
    print(
        f"  PR-AUC (average precision): {ap:.4f}"
        f"  | 무작위 기준선 {base_rate:.4f} 대비 {lift:,.0f}배"
    )
    for k in (100, 500):
        k_eff = min(k, len(scores))
        p = precision_at_k(y_arr, scores, k_eff)
        tp_topk = int(round(p * k_eff))
        rec = tp_topk / n_pos if n_pos > 0 else float("nan")
        print(
            f"  상위 {k_eff}건 조사 시: 정밀도 {p:.3f}"
            f" | 전체 사기의 {rec * 100:.1f}% 회수"
        )


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def build_artifact_bundle(
    model: object,
    *,
    model_version: str,
    default_threshold: float = PAYSIM_DEFAULT_THRESHOLD,
    training_metadata: dict | None = None,
    trained_at: str | None = None,
    library_versions: dict | None = None,
) -> dict:
    """Build a validated, versioned bundle without writing it to disk."""
    predictor = PaySimFraudPredictor(
        model=model,
        feature_names=PAYSIM_FEATURE_NAMES,
        model_name=PAYSIM_MODEL_NAME,
        model_version=model_version,
        default_threshold=default_threshold,
    )
    if training_metadata is not None and not isinstance(training_metadata, dict):
        raise TypeError("training_metadata must be a dictionary or None.")
    if trained_at is not None and not isinstance(trained_at, str):
        raise TypeError("trained_at must be a string or None.")
    if library_versions is not None and not isinstance(library_versions, dict):
        raise TypeError("library_versions must be a dictionary or None.")

    return {
        "artifact_schema_version": PAYSIM_ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": PAYSIM_FEATURE_SCHEMA_VERSION,
        "model": predictor.model,
        "feature_names": predictor.feature_names,
        "model_name": predictor.model_name,
        "model_version": predictor.model_version,
        "default_threshold": predictor.default_threshold,
        "trained_at": trained_at or datetime.now(timezone.utc).isoformat(),
        "training_metadata": dict(training_metadata or {}),
        "library_versions": dict(
            library_versions
            or {
                "python": platform.python_version(),
                "numpy": _installed_version("numpy"),
                "pandas": _installed_version("pandas"),
                "scikit-learn": _installed_version("scikit-learn"),
                "lightgbm": _installed_version("lightgbm"),
                "joblib": _installed_version("joblib"),
            }
        ),
    }


def _dump_artifact(bundle: dict, output_path: Path) -> None:
    import joblib

    joblib.dump(bundle, output_path)


def save_artifact_bundle(
    bundle: dict,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist a validated bundle, refusing overwrite unless explicitly allowed."""
    validated_bundle = validate_paysim_artifact_bundle(bundle)
    if isinstance(output_path, bool) or not isinstance(output_path, (str, Path)):
        raise TypeError("output_path must be a string or pathlib.Path.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool.")

    path = Path(output_path)
    if path.exists() and path.is_dir():
        raise IsADirectoryError(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _dump_artifact(validated_bundle, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="PaySim CSV 경로")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_PAYSIM_MODEL_PATH)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--threshold", type=float, default=PAYSIM_DEFAULT_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("1) 데이터 로드 & Canonical Adapter 변환")
    raw = pd.read_csv(args.csv_path)
    if args.max_step is not None:
        raw = raw[raw["step"] <= args.max_step]
    raw = raw[raw["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
    if TARGET not in raw.columns:
        raise ValueError(f"Training CSV requires target column: {TARGET}")

    print("2) 공통 ML 피처 생성")
    # TODO: 집계 피처는 현재 배치 전체에서 생성된다. 운영 학습에서는 각
    # split의 경계 이후 정보를 앞 구간이 보지 않도록 시점별 생성 정책을 강화한다.
    feat = prepare_paysim_features(raw)
    feat[TARGET] = raw[TARGET].to_numpy()
    print(f"   {len(feat):,}행 | 사기 비율 {feat[TARGET].mean():.4%}")

    print("3) 시간 기반 분할 (train 70% / valid 15% / test 15%)")
    train, valid, test = time_split(feat)
    x_tr, y_tr = train[FEATURES], train[TARGET]
    x_va, y_va = valid[FEATURES], valid[TARGET]
    x_te, y_te = test[FEATURES], test[TARGET]
    base_rate = y_te.mean()

    print("4) 베이스라인: Logistic Regression")
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    logreg.fit(x_tr, y_tr)
    report("Logistic Regression", y_te, logreg.predict_proba(x_te)[:, 1], base_rate)

    print("5) LightGBM")
    pos = max(int(y_tr.sum()), 1)
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        scale_pos_weight=(len(y_tr) - pos) / pos,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    report("LightGBM", y_te, model.predict_proba(x_te)[:, 1], base_rate)

    bundle = build_artifact_bundle(
        model,
        model_version=args.model_version,
        default_threshold=args.threshold,
        training_metadata={
            "target": TARGET,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "test_rows": len(test),
            "max_step": args.max_step,
            "random_state": 42,
        },
    )
    saved_path = save_artifact_bundle(bundle, args.output, overwrite=args.overwrite)
    print(f"6) Artifact 저장: {saved_path}")


if __name__ == "__main__":
    main()
