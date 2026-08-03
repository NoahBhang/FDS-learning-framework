import sys
import builtins
import warnings
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# 프로젝트 루트를 import 경로에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kaggle_bank_fds.src.models import predictor as predictor_module
from kaggle_bank_fds.src.models.paysim_ml_contract import (
    PAYSIM_ARTIFACT_SCHEMA_VERSION,
    PAYSIM_DEFAULT_THRESHOLD,
    PAYSIM_FEATURE_NAMES,
    PAYSIM_FEATURE_SCHEMA_VERSION,
    PAYSIM_MODEL_NAME,
    PAYSIM_REQUIRED_ARTIFACT_KEYS,
)
from kaggle_bank_fds.src.models.predictor import (
    DEFAULT_RULES,
    PAYSIM_PREDICTION_COLUMNS,
    PaySimFraudPredictor,
    predict,
    predict_paysim,
    prepare_paysim_features,
)
from kaggle_bank_fds.src import train as train_module


def make_tx():
    """이체 후 즉시 현금화 + 전액 이체 패턴을 모두 포함한 배치."""
    return pd.DataFrame([
        {
            "step": 1, "type": "TRANSFER", "amount": 100_000,
            "nameOrig": "C1", "oldbalanceOrg": 100_000, "newbalanceOrig": 0,
            "nameDest": "C2", "oldbalanceDest": 0, "newbalanceDest": 100_000,
        },
        {
            "step": 2, "type": "CASH_OUT", "amount": 99_000,
            "nameOrig": "C2", "oldbalanceOrg": 100_000, "newbalanceOrig": 1_000,
            "nameDest": "C3", "oldbalanceDest": 0, "newbalanceDest": 99_000,
        },
    ])


class BrokenRule:
    """예외 격리 테스트용 — 항상 평가에 실패하는 가짜 규칙."""
    rule_name = "broken_rule"

    def evaluate(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_predict_normal_case_returns_expected_shape():
    result = predict(make_tx())

    assert set(result.keys()) == {"fraud_score", "triggered_rules", "details"}
    assert 0.0 <= result["fraud_score"] <= 1.0
    assert "transfer_cash_out" in result["triggered_rules"]
    assert "full_balance_transfer" in result["triggered_rules"]

    # 의심 판정된 규칙은 details에도 반드시 근거(reason·evidence_ids)를 동반한다.
    for rule_name in result["triggered_rules"]:
        detail = result["details"][rule_name]
        assert detail["is_suspicious"] is True
        assert detail["reason"]
        assert len(detail["evidence_ids"]) > 0

    assert result["details"]["skipped_rules"] == {}


def test_predict_ignores_normal_transactions():
    tx = pd.DataFrame([
        {
            "step": 1, "type": "PAYMENT", "amount": 5_000,
            "nameOrig": "C1", "oldbalanceOrg": 100_000, "newbalanceOrig": 95_000,
            "nameDest": "M1", "oldbalanceDest": 0, "newbalanceDest": 5_000,
        },
    ])
    result = predict(tx)

    assert result["fraud_score"] == 0.0
    assert result["triggered_rules"] == []


def test_predict_isolates_failing_rule_and_records_skipped_rules():
    rules = DEFAULT_RULES + [BrokenRule()]
    result = predict(make_tx(), rules=rules)

    # 실패한 규칙은 triggered_rules에서 빠지고 나머지 규칙은 정상 집계된다.
    assert "broken_rule" not in result["triggered_rules"]
    assert "transfer_cash_out" in result["triggered_rules"]
    assert "full_balance_transfer" in result["triggered_rules"]

    # skipped_rules에 실패 사유가 기록된다.
    assert result["details"]["skipped_rules"] == {"broken_rule": "boom"}

    # 성공한 규칙들의 판단은 실패 규칙이 섞여도 영향받지 않는다.
    baseline = predict(make_tx())
    assert result["fraud_score"] == baseline["fraud_score"]


def test_predict_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        predict(pd.DataFrame())


class FakeModel:
    classes_ = np.array([0, 1])

    def __init__(self, positive=None):
        self.positive = positive
        self.seen = None

    def predict_proba(self, matrix):
        self.seen = matrix.copy()
        positive = (
            np.asarray(self.positive, dtype=float)
            if self.positive is not None
            else np.linspace(0.2, 0.8, len(matrix))
        )
        return np.column_stack([1.0 - positive, positive])


def make_paysim_raw(index=None):
    frame = pd.DataFrame(
        [
            {
                "step": 3, "type": "TRANSFER", "amount": 100.0,
                "nameOrig": "C1", "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0,
                "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 100.0,
            },
            {
                "step": 1, "type": "CASH_OUT", "amount": 40.0,
                "nameOrig": "C2", "oldbalanceOrg": 100.0, "newbalanceOrig": 60.0,
                "nameDest": "C3", "oldbalanceDest": 0.0, "newbalanceDest": 40.0,
            },
            {
                "step": 2, "type": "PAYMENT", "amount": 20.0,
                "nameOrig": "C4", "oldbalanceOrg": 50.0, "newbalanceOrig": 30.0,
                "nameDest": "M1", "oldbalanceDest": 10.0, "newbalanceDest": 30.0,
            },
        ]
    )
    if index is not None:
        frame.index = index
    return frame


def make_ml_predictor(model=None, **kwargs):
    return PaySimFraudPredictor(
        model=model or FakeModel(),
        feature_names=kwargs.pop("feature_names", PAYSIM_FEATURE_NAMES),
        model_name=kwargs.pop("model_name", PAYSIM_MODEL_NAME),
        model_version=kwargs.pop("model_version", "test-v1"),
        default_threshold=kwargs.pop("default_threshold", PAYSIM_DEFAULT_THRESHOLD),
        **kwargs,
    )


def valid_bundle(model=None):
    return {
        "artifact_schema_version": PAYSIM_ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": PAYSIM_FEATURE_SCHEMA_VERSION,
        "model": model or FakeModel(),
        "feature_names": PAYSIM_FEATURE_NAMES,
        "model_name": PAYSIM_MODEL_NAME,
        "model_version": "test-v1",
        "default_threshold": PAYSIM_DEFAULT_THRESHOLD,
    }


def test_ml_contract_constants_are_immutable_and_complete():
    assert isinstance(PAYSIM_FEATURE_NAMES, tuple)
    assert isinstance(PAYSIM_REQUIRED_ARTIFACT_KEYS, frozenset)
    assert len(PAYSIM_FEATURE_NAMES) == 11
    assert PAYSIM_REQUIRED_ARTIFACT_KEYS <= valid_bundle().keys()


def test_predictor_constructor_normalizes_metadata_and_features():
    instance = make_ml_predictor(
        feature_names=[" log_amount "], model_name=" model ", model_version=" v1 "
    )
    assert instance.feature_names == ("log_amount",)
    assert instance.model_name == "model"
    assert instance.model_version == "v1"


@pytest.mark.parametrize("model", [None, object()])
def test_predictor_constructor_rejects_invalid_model(model):
    with pytest.raises(TypeError):
        PaySimFraudPredictor(
            model=model,
            feature_names=PAYSIM_FEATURE_NAMES,
            model_name=PAYSIM_MODEL_NAME,
            model_version="test-v1",
        )


@pytest.mark.parametrize("features", [[], ["x", "x"], "x", [1], [" "]])
def test_predictor_constructor_rejects_invalid_feature_names(features):
    with pytest.raises((TypeError, ValueError)):
        make_ml_predictor(feature_names=features)


@pytest.mark.parametrize("field,value", [("model_name", ""), ("model_version", " ")])
def test_predictor_constructor_rejects_empty_metadata(field, value):
    with pytest.raises(ValueError):
        make_ml_predictor(**{field: value})


@pytest.mark.parametrize("threshold", [True, False, "0.5", None])
def test_predictor_rejects_non_real_thresholds(threshold):
    with pytest.raises(TypeError):
        make_ml_predictor(default_threshold=threshold)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_predictor_rejects_out_of_range_or_nonfinite_thresholds(threshold):
    with pytest.raises(ValueError):
        make_ml_predictor(default_threshold=threshold)


@pytest.mark.parametrize("threshold", [0, 0.25, np.float32(0.5), 1])
def test_predictor_accepts_threshold_boundaries_and_numpy_real(threshold):
    assert make_ml_predictor(default_threshold=threshold).default_threshold == float(threshold)


def test_prepare_features_uses_contract_and_restores_source_order():
    raw = make_paysim_raw(index=[30, 10, 20])
    features = prepare_paysim_features(raw)
    assert tuple(features.loc[:, PAYSIM_FEATURE_NAMES].columns) == PAYSIM_FEATURE_NAMES
    assert features.index.tolist() == [30, 10, 20]
    assert features["source_row_id"].tolist() == [0, 1, 2]


def test_prepare_features_does_not_mutate_input():
    raw = make_paysim_raw(index=[30, 10, 20])
    before = raw.copy(deep=True)
    prepare_paysim_features(raw)
    assert_frame_equal(raw, before)


@pytest.mark.parametrize(
    "column,value",
    [("step", np.nan), ("oldbalanceOrg", np.inf), ("newbalanceDest", -np.inf)],
)
def test_prepare_features_rejects_missing_or_nonfinite_numeric_values(column, value):
    raw = make_paysim_raw()
    raw.loc[0, column] = value
    with pytest.raises(ValueError, match="finite"):
        prepare_paysim_features(raw)


def test_predict_returns_exact_schema_dtypes_and_original_index():
    raw = make_paysim_raw(index=[30, 10, 20])
    result = make_ml_predictor(FakeModel([0.1, 0.5, 0.9])).predict(raw)
    assert tuple(result.columns) == PAYSIM_PREDICTION_COLUMNS
    assert result.index.tolist() == raw.index.tolist()
    assert [str(dtype) for dtype in result.dtypes] == [
        "string", "Int64", "Float64", "boolean", "Float64", "string", "string"
    ]
    assert result["predicted_fraud"].tolist() == [False, True, True]
    assert result["model_name"].unique().tolist() == [PAYSIM_MODEL_NAME]
    assert result["model_version"].unique().tolist() == ["test-v1"]


def test_predict_accepts_probability_boundaries_zero_and_one():
    result = make_ml_predictor(FakeModel([0.0, 0.5, 1.0])).predict(make_paysim_raw())
    assert result["fraud_probability"].tolist() == [0.0, 0.5, 1.0]
    assert result["predicted_fraud"].tolist() == [False, True, True]


def test_predict_passes_finite_feature_matrix_in_contract_order():
    model = FakeModel()
    make_ml_predictor(model).predict(make_paysim_raw())
    assert tuple(model.seen.columns) == PAYSIM_FEATURE_NAMES
    assert np.isfinite(model.seen.to_numpy()).all()


def test_predict_does_not_mutate_input():
    raw = make_paysim_raw()
    before = raw.copy(deep=True)
    make_ml_predictor().predict(raw)
    assert_frame_equal(raw, before)


def test_predict_valid_empty_frame_returns_typed_empty_without_model_call():
    raw = make_paysim_raw().iloc[:0]
    model = FakeModel()
    result = make_ml_predictor(model).predict(raw)
    assert result.empty
    assert tuple(result.columns) == PAYSIM_PREDICTION_COLUMNS
    assert model.seen is None


def test_predict_rejects_non_dataframe():
    with pytest.raises(TypeError):
        make_ml_predictor().predict([])


def test_predict_rejects_missing_raw_column():
    with pytest.raises(ValueError):
        make_ml_predictor().predict(make_paysim_raw().drop(columns="amount"))


def test_predict_rejects_duplicate_canonical_transaction_ids(monkeypatch):
    original = predictor_module.PaySimAdapter.transform

    def duplicated(adapter, frame):
        canonical = original(adapter, frame)
        canonical.loc[canonical.index[1], "transaction_id"] = canonical.iloc[0]["transaction_id"]
        return canonical

    monkeypatch.setattr(predictor_module.PaySimAdapter, "transform", duplicated)
    with pytest.raises(ValueError, match="unique"):
        make_ml_predictor().predict(make_paysim_raw())


class BadProbabilityModel(FakeModel):
    def __init__(self, output, classes=None):
        super().__init__()
        self.output = output
        if classes is not None:
            self.classes_ = np.asarray(classes)

    def predict_proba(self, matrix):
        return self.output


@pytest.mark.parametrize(
    "output",
    [np.array([0.1, 0.2]), np.ones((2, 2)), np.ones((3, 1))],
)
def test_predict_rejects_invalid_predict_proba_shapes(output):
    with pytest.raises(ValueError):
        make_ml_predictor(BadProbabilityModel(output)).predict(make_paysim_raw())


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_predict_rejects_invalid_positive_probabilities(value):
    output = np.array([[0.9, value], [0.8, 0.2], [0.7, 0.3]])
    with pytest.raises(ValueError):
        make_ml_predictor(BadProbabilityModel(output)).predict(make_paysim_raw())


def test_predict_uses_classes_to_locate_positive_probability():
    model = BadProbabilityModel(
        np.array([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]]), classes=[1, 0]
    )
    result = make_ml_predictor(model).predict(make_paysim_raw())
    assert result["fraud_probability"].tolist() == [0.8, 0.5, 0.1]


@pytest.mark.parametrize("classes", [[0, 2], [0, 1, 2], [1, 1]])
def test_predict_rejects_invalid_classes_contract(classes):
    output = np.ones((3, 2)) * 0.5
    with pytest.raises(ValueError):
        make_ml_predictor(BadProbabilityModel(output, classes=classes)).predict(
            make_paysim_raw()
        )


def test_predict_without_classes__uses_second_binary_column():
    class ModelWithoutClasses:
        def predict_proba(self, matrix):
            return np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])

    model = ModelWithoutClasses()
    result = make_ml_predictor(model).predict(make_paysim_raw())
    assert result["fraud_probability"].tolist() == [0.1, 0.6, 0.8]


def test_predict_without_classes_rejects_multiclass_output():
    class ModelWithoutClasses:
        def predict_proba(self, matrix):
            return np.ones((3, 3)) / 3

    model = ModelWithoutClasses()
    with pytest.raises(ValueError):
        make_ml_predictor(model).predict(make_paysim_raw())


def test_predict_threshold_override_is_applied():
    result = make_ml_predictor(FakeModel([0.6, 0.7, 0.8])).predict(
        make_paysim_raw(), threshold=0.7
    )
    assert result["threshold"].tolist() == [0.7, 0.7, 0.7]
    assert result["predicted_fraud"].tolist() == [False, True, True]


def test_predict_paysim_rejects_predictor_and_model_path_together(tmp_path):
    with pytest.raises(ValueError):
        predict_paysim(make_paysim_raw(), predictor=make_ml_predictor(), model_path=tmp_path)


def test_predict_paysim_uses_injected_predictor_without_loading(monkeypatch):
    def fail_load(*args):
        raise AssertionError("artifact must not be loaded")

    monkeypatch.setattr(PaySimFraudPredictor, "from_artifact", fail_load)
    result = predict_paysim(make_paysim_raw(), predictor=make_ml_predictor())
    assert len(result) == 3


def test_predict_paysim_preserves_injected_predictor_default_threshold():
    instance = make_ml_predictor(
        FakeModel([0.6, 0.8, 0.9]), default_threshold=0.8
    )
    result = predict_paysim(make_paysim_raw(), predictor=instance)
    assert result["threshold"].tolist() == [0.8, 0.8, 0.8]
    assert result["predicted_fraud"].tolist() == [False, True, True]


def test_predict_paysim_explicit_threshold_overrides_predictor_default():
    instance = make_ml_predictor(
        FakeModel([0.6, 0.8, 0.9]), default_threshold=0.8
    )
    result = predict_paysim(
        make_paysim_raw(), predictor=instance, threshold=0.5
    )
    assert result["threshold"].tolist() == [0.5, 0.5, 0.5]
    assert result["predicted_fraud"].tolist() == [True, True, True]


def test_predict_paysim_preserves_loaded_artifact_threshold(monkeypatch, tmp_path):
    instance = make_ml_predictor(
        FakeModel([0.6, 0.8, 0.9]), default_threshold=0.8
    )
    monkeypatch.setattr(
        PaySimFraudPredictor,
        "from_artifact",
        classmethod(lambda cls, path: instance),
    )
    result = predict_paysim(make_paysim_raw(), model_path=tmp_path / "model.joblib")
    assert result["threshold"].tolist() == [0.8, 0.8, 0.8]
    assert result["predicted_fraud"].tolist() == [False, True, True]


def test_predict_paysim_loads_path_lazily(monkeypatch, tmp_path):
    path = tmp_path / "model.joblib"
    seen = []

    def fake_load(cls, model_path):
        seen.append(model_path)
        return make_ml_predictor()

    monkeypatch.setattr(PaySimFraudPredictor, "from_artifact", classmethod(fake_load))
    predict_paysim(make_paysim_raw(), model_path=path)
    assert seen == [path]


@pytest.mark.parametrize("path_value", [None, 1, True])
def test_from_artifact_rejects_invalid_path_type(path_value):
    with pytest.raises(TypeError):
        PaySimFraudPredictor.from_artifact(path_value)


def test_from_artifact_reports_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        PaySimFraudPredictor.from_artifact(tmp_path / "missing.joblib")


def test_from_artifact_rejects_directory(tmp_path):
    with pytest.raises(IsADirectoryError):
        PaySimFraudPredictor.from_artifact(tmp_path)


def load_bundle(monkeypatch, tmp_path, bundle):
    path = tmp_path / "artifact.joblib"
    path.touch()
    monkeypatch.setattr(predictor_module, "_load_artifact", lambda _: bundle)
    return path


def test_from_artifact_accepts_valid_contract(monkeypatch, tmp_path):
    instance = PaySimFraudPredictor.from_artifact(
        load_bundle(monkeypatch, tmp_path, valid_bundle())
    )
    assert instance.feature_names == PAYSIM_FEATURE_NAMES


def test_from_artifact_rejects_non_dict(monkeypatch, tmp_path):
    with pytest.raises(TypeError):
        PaySimFraudPredictor.from_artifact(load_bundle(monkeypatch, tmp_path, []))


def test_from_artifact_rejects_missing_keys(monkeypatch, tmp_path):
    bundle = valid_bundle()
    del bundle["model_version"]
    with pytest.raises(ValueError, match="missing keys"):
        PaySimFraudPredictor.from_artifact(load_bundle(monkeypatch, tmp_path, bundle))


@pytest.mark.parametrize("key", ["artifact_schema_version", "feature_schema_version"])
def test_from_artifact_rejects_schema_version_mismatch(monkeypatch, tmp_path, key):
    bundle = valid_bundle()
    bundle[key] = "999"
    with pytest.raises(ValueError, match="schema version"):
        PaySimFraudPredictor.from_artifact(load_bundle(monkeypatch, tmp_path, bundle))


def test_from_artifact_rejects_feature_contract_mismatch(monkeypatch, tmp_path):
    bundle = valid_bundle()
    bundle["feature_names"] = tuple(reversed(PAYSIM_FEATURE_NAMES))
    with pytest.raises(ValueError, match="feature_names"):
        PaySimFraudPredictor.from_artifact(load_bundle(monkeypatch, tmp_path, bundle))


def test_from_artifact_rejects_invalid_model(monkeypatch, tmp_path):
    bundle = valid_bundle()
    bundle["model"] = object()
    with pytest.raises(TypeError, match="predict_proba"):
        PaySimFraudPredictor.from_artifact(load_bundle(monkeypatch, tmp_path, bundle))


@pytest.mark.parametrize(
    "values",
    [
        np.array([[False, True], [True, False], [False, True]], dtype=bool),
        np.array([[0, True], [1, False], [0, np.bool_(True)]], dtype=object),
        np.array([[0.4, 0.6], [0.3, 0.7], [0.2, 0.8]], dtype=np.complex64),
        np.array([[0.4, 0.6], [0.3, 0.7], [0.2, 0.8]], dtype=np.complex128),
        np.array([[0.4, 0.6 + 1j], [0.3, 0.7], [0.2, 0.8]], dtype=object),
        np.array([["0.4", "0.6"], ["0.3", "0.7"], ["0.2", "0.8"]]),
        np.array([[0.4, "0.6"], [0.3, 0.7], [0.2, 0.8]], dtype=object),
        np.array(
            [[Decimal("0.4"), Decimal("0.6")]] * 3,
            dtype=object,
        ),
    ],
    ids=[
        "bool-dtype",
        "python-and-numpy-bool-object",
        "complex64",
        "complex128",
        "complex-object",
        "numeric-string",
        "mixed-object",
        "decimal-object",
    ],
)
def test_predict_rejects_non_real_probability_dtypes_without_warning(values):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="real integer or floating"):
            make_ml_predictor(BadProbabilityModel(values)).predict(make_paysim_raw())
    assert caught == []


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int64, np.float32, np.float64],
)
def test_predict_accepts_real_integer_and_floating_probability_dtypes(dtype):
    values = np.array([[1, 0], [0, 1], [1, 0]], dtype=dtype)
    result = make_ml_predictor(BadProbabilityModel(values)).predict(make_paysim_raw())
    assert result["fraud_probability"].tolist() == [0.0, 1.0, 0.0]


def test_artifact_loader_reports_missing_optional_dependency(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def missing_joblib(name, *args, **kwargs):
        if name == "joblib":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_joblib)
    with pytest.raises(ImportError, match="optional ML dependencies"):
        predictor_module._load_artifact(tmp_path / "unused.joblib")


def test_build_artifact_bundle_has_contract_and_metadata():
    bundle = train_module.build_artifact_bundle(
        FakeModel(),
        model_version="v2",
        training_metadata={"rows": 3},
        trained_at="2026-01-01T00:00:00+00:00",
        library_versions={"python": "test"},
    )
    assert PAYSIM_REQUIRED_ARTIFACT_KEYS <= bundle.keys()
    assert bundle["feature_names"] == PAYSIM_FEATURE_NAMES
    assert bundle["training_metadata"] == {"rows": 3}
    assert bundle["trained_at"] == "2026-01-01T00:00:00+00:00"


@pytest.mark.parametrize("metadata", [[], "bad", 1])
def test_build_artifact_bundle_rejects_invalid_training_metadata(metadata):
    with pytest.raises(TypeError):
        train_module.build_artifact_bundle(
            FakeModel(), model_version="v1", training_metadata=metadata
        )


def test_save_artifact_bundle_does_not_write_without_dump_call(monkeypatch, tmp_path):
    bundle = valid_bundle()
    output = tmp_path / "models" / "artifact.joblib"
    seen = []
    monkeypatch.setattr(train_module, "_dump_artifact", lambda b, p: seen.append((b, p)))
    assert train_module.save_artifact_bundle(bundle, output) == output
    assert seen == [(bundle, output)]


def test_save_artifact_bundle_refuses_overwrite(monkeypatch, tmp_path):
    output = tmp_path / "artifact.joblib"
    output.touch()
    monkeypatch.setattr(train_module, "_dump_artifact", lambda *args: None)
    with pytest.raises(FileExistsError):
        train_module.save_artifact_bundle(valid_bundle(), output)


def test_save_artifact_bundle_allows_explicit_overwrite(monkeypatch, tmp_path):
    output = tmp_path / "artifact.joblib"
    output.touch()
    seen = []
    monkeypatch.setattr(train_module, "_dump_artifact", lambda b, p: seen.append(p))
    train_module.save_artifact_bundle(valid_bundle(), output, overwrite=True)
    assert seen == [output]


def test_predictor_and_train_share_bundle_validator():
    assert (
        predictor_module.validate_paysim_artifact_bundle
        is train_module.validate_paysim_artifact_bundle
    )


def test_bundle_validator_does_not_mutate_bundle():
    bundle = valid_bundle()
    before = bundle.copy()
    assert predictor_module.validate_paysim_artifact_bundle(bundle) is bundle
    assert bundle == before


@pytest.mark.parametrize(
    "mutate,exception",
    [
        (lambda bundle: bundle.pop("model_version"), ValueError),
        (lambda bundle: bundle.__setitem__("artifact_schema_version", "999"), ValueError),
        (lambda bundle: bundle.__setitem__("feature_schema_version", "999"), ValueError),
        (lambda bundle: bundle.__setitem__("feature_names", ("wrong",)), ValueError),
        (lambda bundle: bundle.__setitem__("feature_names", list(PAYSIM_FEATURE_NAMES)), ValueError),
        (lambda bundle: bundle.__setitem__("model", object()), TypeError),
        (lambda bundle: bundle.__setitem__("default_threshold", True), TypeError),
        (lambda bundle: bundle.__setitem__("default_threshold", 1.1), ValueError),
        (lambda bundle: bundle.__setitem__("trained_at", ""), ValueError),
        (lambda bundle: bundle.__setitem__("training_metadata", []), TypeError),
        (lambda bundle: bundle.__setitem__("library_versions", []), TypeError),
    ],
    ids=[
        "missing-key",
        "artifact-schema",
        "feature-schema",
        "features",
        "features-not-tuple",
        "model",
        "threshold-bool",
        "threshold-range",
        "trained-at",
        "training-metadata",
        "library-versions",
    ],
)
def test_save_rejects_invalid_bundle_before_dump_or_file_creation(
    monkeypatch, tmp_path, mutate, exception
):
    bundle = valid_bundle()
    mutate(bundle)
    output = tmp_path / "new-parent" / "artifact.joblib"
    dump_calls = []
    monkeypatch.setattr(
        train_module,
        "_dump_artifact",
        lambda *args: dump_calls.append(args),
    )

    with pytest.raises(exception):
        train_module.save_artifact_bundle(bundle, output)

    assert dump_calls == []
    assert not output.exists()
    assert not output.parent.exists()


def test_training_bundle_can_be_loaded_by_predictor(monkeypatch, tmp_path):
    bundle = train_module.build_artifact_bundle(FakeModel(), model_version="roundtrip-v1")
    instance = PaySimFraudPredictor.from_artifact(
        load_bundle(monkeypatch, tmp_path, deepcopy(bundle))
    )
    result = instance.predict(make_paysim_raw())
    assert result["model_version"].unique().tolist() == ["roundtrip-v1"]
