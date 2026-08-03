"""
kaggle_bank_fds/src/models/predictor.py
============================================================
predict() — DB/학습 파이프라인 없이 거래 DataFrame 하나로 즉시 판단을 돌려주는 진입점

[bankruptcy_fds/src/models.py와의 관계]
파산관재인 FDS 쪽 predict()와 반환 형태·예외 처리 방식이 동일하다 —
"같은 사고틀, 다른 도메인"이라는 이 저장소의 핵심 메시지를 서빙
계층에서도 반복하는 것이다. 차이는 도메인 규칙뿐이다:
파산 FDS는 파산자 1명 단위(LayeringRule의 BFS가 출발 노드를 요구)로
스코어링하지만, 은행권 FDS(PaySim)의 두 규칙은 self-merge/전체 집계
기반이라 특정 계좌 하나로 입력을 제한할 필요가 없다 — 배치 전체를
한 번에 평가한다.

판단 로직은 새로 만들지 않는다. rules/bank_fraud_rules.py의 두 규칙
(BaseFraudRule 하위 클래스)과 shared.scoring.RiskScorer를 그대로
재사용한다.
"""

import logging
import math
import sys
from collections.abc import Sequence
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[3]))

from shared.scoring.risk_scorer import RiskScorer                    # noqa: E402
from kaggle_bank_fds.src.rules.bank_fraud_rules import (             # noqa: E402
    TransferCashOutRule, FullBalanceTransferRule,
)
from kaggle_bank_fds.src.adapters.paysim_adapter import PaySimAdapter  # noqa: E402
from kaggle_bank_fds.src.models.paysim_ml_contract import (           # noqa: E402
    PAYSIM_DEFAULT_THRESHOLD,
    PAYSIM_FEATURE_NAMES,
    validate_paysim_artifact_bundle,
)
from shared.features import add_aggregate_features, add_raw_features  # noqa: E402

logger = logging.getLogger(__name__)

# 기본 규칙 세트 — 둘 다 배치 전체(여러 계좌)를 대상으로 self-merge/집계한다.
DEFAULT_RULES = [TransferCashOutRule(), FullBalanceTransferRule()]


def predict(transaction_data: pd.DataFrame, rules: list = None) -> dict:
    """
    은행 거래 배치(PaySim 스키마)를 규칙 세트로 평가해 종합 판단을 반환한다.

    Parameters
    ----------
    transaction_data : PaySim 원본 컬럼(type, amount, nameOrig, nameDest,
                        oldbalanceOrg, step, ...)을 가진 DataFrame.
                        bankruptcy_fds.src.models.predict()와 달리 단일
                        엔티티로 제한하지 않는다 — TransferCashOutRule이
                        계좌 간 이체→인출 연쇄를 self-merge로 찾아내려면
                        여러 계좌가 섞인 배치 전체가 필요하기 때문이다.
                        원본 인덱스를 보존해야 한다: PaySim에는 거래 고유
                        ID 컬럼이 없어 두 규칙 모두 evidence_ids로
                        DataFrame의 행 인덱스를 그대로 쓴다.
    rules : 평가에 쓸 규칙 목록. 생략 시 DEFAULT_RULES(이체 후 즉시
            현금화, 계좌 전액 이체) 전부를 쓴다. bankruptcy_fds 쪽과
            같은 DI 방식 — 테스트에서 규칙 하나만 주입해 검증할 때
            쓰는 확장점이다.

    Returns
    -------
    dict
        fraud_score      : 0.0~1.0 종합 위험 점수 (RiskScorer의 0~100점을
                            정규화). 평가에 실패한 규칙은 집계에서 제외된다.
        triggered_rules  : 의심 판정된 규칙 이름(rule_name) 목록.
                            평가에 실패한 규칙은 포함되지 않는다.
        details          : 규칙별 상세 판단 +
                            {rule_name: {risk_type, is_suspicious, risk_score,
                                         reason, evidence_ids}, ...,
                             "skipped_rules": {rule_name: 에러 메시지, ...}}

    알려진 제약
    -----------
    - bankruptcy_fds.src.models.predict()와 달리 단일 엔티티로 입력을
      제한하지 않는다. 배치에 여러 계좌가 섞여도 정상 동작하지만,
      그만큼 이 함수는 "배치 전체에 대한 종합 판단" 하나만 반환한다 —
      계좌별로 보려면 호출 전에 직접 계좌 단위로 나눠 여러 번 호출해야
      한다.
    - PaySim 데이터에는 거래 고유 ID가 없어 evidence_ids는 DataFrame
      행 인덱스다. 인덱스를 초기화(reset_index)한 복사본을 넘기면
      근거 거래가 원본과 어긋난다.
    - 규칙 하나가 예외를 던지면 그 규칙은 조용히 건너뛰고 나머지
      규칙으로 계속 진행한다(가용성 우선). 그 결과 fraud_score가
      실제보다 낮게 나올 수 있다 — 반환된 details["skipped_rules"]가
      비어 있지 않다면 그 판단은 불완전하다는 뜻이므로 반드시 확인해야
      한다.
    - 라벨(isFraud/isFlaggedFraud)은 채점 전용이며 이 함수는 참조하지
      않는다 — 규칙이 실서비스에서 라벨 없이 동작해야 한다는 전제를
      지킨다.
    """
    if transaction_data.empty:
        raise ValueError("transaction_data가 비어 있다.")

    active_rules = rules if rules is not None else DEFAULT_RULES

    # 규칙 하나의 실패가 전체 판단을 막지 않도록 개별적으로 감싼다.
    # (bankruptcy_fds.src.models.predict()와 동일한 방어 방식)
    results = []
    skipped_rules = {}
    for rule in active_rules:
        try:
            results.append(rule.evaluate(transaction_data))
        except Exception as exc:
            logger.error("규칙 '%s' 평가 실패, 건너뛴다: %s", rule.rule_name, exc)
            skipped_rules[rule.rule_name] = str(exc)

    report = RiskScorer().aggregate("batch", results)

    details = {
        r.rule_name: {
            "risk_type": r.risk_type,
            "is_suspicious": r.is_suspicious,
            "risk_score": r.risk_score,
            "reason": r.reason,
            "evidence_ids": r.evidence_ids,
        }
        for r in results
    }
    details["skipped_rules"] = skipped_rules

    return {
        "fraud_score": report.total_score / 100.0,
        "triggered_rules": [r.rule_name for r in report.findings],
        "details": details,
    }


PAYSIM_BASE_TIMESTAMP = pd.Timestamp("2026-01-01")
DEFAULT_PAYSIM_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "paysim_lightgbm.joblib"
)
PAYSIM_PREDICTION_COLUMNS = (
    "transaction_id",
    "source_row_id",
    "fraud_probability",
    "predicted_fraud",
    "threshold",
    "model_name",
    "model_version",
)


def _validate_threshold(value: object, *, field_name: str = "threshold") -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1.")
    return normalized


def _validate_nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _validate_feature_names(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("feature_names must be a sequence of strings.")
    normalized = tuple(
        _validate_nonempty_string(item, field_name="feature_names item")
        for item in value
    )
    if not normalized:
        raise ValueError("feature_names must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("feature_names must be unique.")
    return normalized


def _empty_prediction_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": pd.Series(pd.array([], dtype="string"), index=index),
            "source_row_id": pd.Series(pd.array([], dtype="Int64"), index=index),
            "fraud_probability": pd.Series(pd.array([], dtype="Float64"), index=index),
            "predicted_fraud": pd.Series(pd.array([], dtype="boolean"), index=index),
            "threshold": pd.Series(pd.array([], dtype="Float64"), index=index),
            "model_name": pd.Series(pd.array([], dtype="string"), index=index),
            "model_version": pd.Series(pd.array([], dtype="string"), index=index),
        },
        columns=PAYSIM_PREDICTION_COLUMNS,
        index=index,
    )


def prepare_paysim_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw PaySim rows to the feature frame shared by train and predict."""
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("raw_df must be a pandas DataFrame.")

    canonical = PaySimAdapter().transform(raw_df)
    if canonical.empty:
        return canonical.copy()
    if canonical["transaction_id"].duplicated().any():
        raise ValueError("Canonical transaction_id values must be unique.")

    numeric_columns = (
        "step",
        "amount",
        "balance_before",
        "balance_after",
        "target_balance_before",
        "target_balance_after",
    )
    for column in numeric_columns:
        values = pd.to_numeric(canonical[column], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Canonical {column} must contain only finite values.")

    features = canonical.copy()
    features["timestamp"] = PAYSIM_BASE_TIMESTAMP + pd.to_timedelta(
        features["step"].astype("int64"), unit="h"
    )
    features["balance_error_orig"] = (
        features["balance_before"]
        - features["amount"]
        - features["balance_after"]
    )
    features["balance_error_dest"] = (
        features["target_balance_before"]
        + features["amount"]
        - features["target_balance_after"]
    )
    features["orig_emptied"] = (
        features["balance_before"].gt(0)
        & features["balance_after"].eq(0)
    ).astype("int64")
    features["is_transfer"] = features["action_type"].eq("TRANSFER").astype("int64")
    features = add_raw_features(features)
    features = add_aggregate_features(features)
    features = features.sort_values("source_row_id", kind="stable")
    features.index = raw_df.index
    return features


def _load_artifact(path: Path) -> object:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "PaySim ML artifact loading requires the optional ML dependencies."
        ) from exc

    return joblib.load(path)


class PaySimFraudPredictor:
    """Validated inference API for a preloaded PaySim binary classifier."""

    def __init__(
        self,
        *,
        model: object,
        feature_names: Sequence[str],
        model_name: str,
        model_version: str,
        default_threshold: float = PAYSIM_DEFAULT_THRESHOLD,
    ) -> None:
        if model is None or not callable(getattr(model, "predict_proba", None)):
            raise TypeError("model must provide a callable predict_proba method.")
        self.model = model
        self.feature_names = _validate_feature_names(feature_names)
        self.model_name = _validate_nonempty_string(model_name, field_name="model_name")
        self.model_version = _validate_nonempty_string(
            model_version, field_name="model_version"
        )
        self.default_threshold = _validate_threshold(
            default_threshold, field_name="default_threshold"
        )

    @classmethod
    def from_artifact(cls, model_path: str | Path) -> "PaySimFraudPredictor":
        """Load a trusted joblib bundle; never load artifacts from untrusted sources."""
        if isinstance(model_path, bool) or not isinstance(model_path, (str, Path)):
            raise TypeError("model_path must be a string or pathlib.Path.")
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            raise IsADirectoryError(path)

        bundle = validate_paysim_artifact_bundle(_load_artifact(path))
        artifact_features = tuple(bundle["feature_names"])

        return cls(
            model=bundle["model"],
            feature_names=artifact_features,
            model_name=bundle["model_name"],
            model_version=bundle["model_version"],
            default_threshold=bundle["default_threshold"],
        )

    def predict(
        self,
        raw_df: pd.DataFrame,
        *,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        if not isinstance(raw_df, pd.DataFrame):
            raise TypeError("raw_df must be a pandas DataFrame.")
        effective_threshold = (
            self.default_threshold
            if threshold is None
            else _validate_threshold(threshold)
        )
        features = prepare_paysim_features(raw_df)
        if features.empty:
            return _empty_prediction_frame(raw_df.index)

        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            raise ValueError(f"Missing model features: {sorted(missing_features)}")
        matrix = features.loc[:, self.feature_names].astype(float)
        matrix_values = matrix.to_numpy(dtype=float)
        if not np.isfinite(matrix_values).all():
            raise ValueError("Model features must contain only finite values.")

        probabilities = np.asarray(self.model.predict_proba(matrix))
        if probabilities.ndim != 2:
            raise ValueError("predict_proba must return a two-dimensional array.")
        if probabilities.shape[0] != len(raw_df):
            raise ValueError("predict_proba row count does not match the input.")
        if probabilities.shape[1] < 2:
            raise ValueError("predict_proba must return at least two class columns.")

        classes = getattr(self.model, "classes_", None)
        if classes is None:
            if probabilities.shape[1] != 2:
                raise ValueError("A model without classes_ must have two output columns.")
            positive_column = 1
        else:
            class_values = np.asarray(classes)
            if class_values.ndim != 1 or len(class_values) != probabilities.shape[1]:
                raise ValueError("model.classes_ does not match predict_proba columns.")
            matches = np.flatnonzero(class_values == 1)
            if len(matches) != 1:
                raise ValueError("model.classes_ must contain the positive class label 1 once.")
            positive_column = int(matches[0])

        raw_probability = np.asarray(probabilities[:, positive_column])
        if not (
            np.issubdtype(raw_probability.dtype, np.integer)
            or np.issubdtype(raw_probability.dtype, np.floating)
        ):
            raise TypeError(
                "Fraud probabilities must contain only real integer or floating values."
            )
        fraud_probability = raw_probability.astype(float, copy=False)
        if not np.isfinite(fraud_probability).all():
            raise ValueError("Fraud probabilities must be finite.")
        if ((fraud_probability < 0) | (fraud_probability > 1)).any():
            raise ValueError("Fraud probabilities must be between 0 and 1.")

        index = raw_df.index
        return pd.DataFrame(
            {
                "transaction_id": pd.Series(
                    features["transaction_id"].array, index=index, dtype="string"
                ),
                "source_row_id": pd.Series(
                    features["source_row_id"].array, index=index, dtype="Int64"
                ),
                "fraud_probability": pd.Series(
                    pd.array(fraud_probability, dtype="Float64"), index=index
                ),
                "predicted_fraud": pd.Series(
                    pd.array(fraud_probability >= effective_threshold, dtype="boolean"),
                    index=index,
                ),
                "threshold": pd.Series(
                    pd.array([effective_threshold] * len(raw_df), dtype="Float64"),
                    index=index,
                ),
                "model_name": pd.Series(
                    pd.array([self.model_name] * len(raw_df), dtype="string"), index=index
                ),
                "model_version": pd.Series(
                    pd.array([self.model_version] * len(raw_df), dtype="string"),
                    index=index,
                ),
            },
            columns=PAYSIM_PREDICTION_COLUMNS,
            index=index,
        )


def predict_paysim(
    raw_df: pd.DataFrame,
    *,
    threshold: float | None = None,
    model_path: str | Path | None = None,
    predictor: PaySimFraudPredictor | None = None,
) -> pd.DataFrame:
    """Predict PaySim rows; loading by path intentionally reloads on every call."""
    if predictor is not None and model_path is not None:
        raise ValueError("Provide either predictor or model_path, not both.")
    if predictor is not None:
        if not isinstance(predictor, PaySimFraudPredictor):
            raise TypeError("predictor must be a PaySimFraudPredictor.")
        return predictor.predict(raw_df, threshold=threshold)

    resolved_path = DEFAULT_PAYSIM_MODEL_PATH if model_path is None else model_path
    loaded_predictor = PaySimFraudPredictor.from_artifact(resolved_path)
    return loaded_predictor.predict(raw_df, threshold=threshold)
