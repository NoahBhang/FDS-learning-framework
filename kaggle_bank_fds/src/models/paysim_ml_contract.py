"""PaySim ML artifact and feature contracts shared by training and inference."""

import math
from numbers import Real

import numpy as np

PAYSIM_ARTIFACT_SCHEMA_VERSION = "1"
PAYSIM_FEATURE_SCHEMA_VERSION = "1"
PAYSIM_MODEL_NAME = "LightGBM"
PAYSIM_DEFAULT_THRESHOLD = 0.5

PAYSIM_FEATURE_NAMES = (
    "log_amount",
    "hour",
    "is_night",
    "is_weekend",
    "recipient_tx_count_30d",
    "same_day_recipient_count",
    "same_day_recipient_total",
    "balance_error_orig",
    "balance_error_dest",
    "orig_emptied",
    "is_transfer",
)

PAYSIM_REQUIRED_ARTIFACT_KEYS = frozenset(
    {
        "artifact_schema_version",
        "feature_schema_version",
        "model",
        "feature_names",
        "model_name",
        "model_version",
        "default_threshold",
    }
)


def validate_paysim_artifact_bundle(bundle: object) -> dict[str, object]:
    """Validate a PaySim artifact bundle without mutating it."""
    if not isinstance(bundle, dict):
        raise TypeError("PaySim artifact must be a dictionary bundle.")

    missing = PAYSIM_REQUIRED_ARTIFACT_KEYS - bundle.keys()
    if missing:
        raise ValueError(f"PaySim artifact is missing keys: {sorted(missing)}")
    if bundle["artifact_schema_version"] != PAYSIM_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported PaySim artifact schema version.")
    if bundle["feature_schema_version"] != PAYSIM_FEATURE_SCHEMA_VERSION:
        raise ValueError("Unsupported PaySim feature schema version.")

    feature_names = bundle["feature_names"]
    if not isinstance(feature_names, tuple) or feature_names != PAYSIM_FEATURE_NAMES:
        raise ValueError("Artifact feature_names do not match the PaySim contract.")

    model = bundle["model"]
    if model is None or not callable(getattr(model, "predict_proba", None)):
        raise TypeError("Artifact model must provide a callable predict_proba method.")

    for key in ("model_name", "model_version"):
        value = bundle[key]
        if not isinstance(value, str):
            raise TypeError(f"Artifact {key} must be a string.")
        if not value.strip():
            raise ValueError(f"Artifact {key} must not be empty.")

    threshold = bundle["default_threshold"]
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise TypeError("Artifact default_threshold must be a real number.")
    normalized_threshold = float(threshold)
    if not math.isfinite(normalized_threshold) or not 0.0 <= normalized_threshold <= 1.0:
        raise ValueError(
            "Artifact default_threshold must be finite and between 0 and 1."
        )

    if "trained_at" in bundle:
        trained_at = bundle["trained_at"]
        if not isinstance(trained_at, str):
            raise TypeError("Artifact trained_at must be a string.")
        if not trained_at.strip():
            raise ValueError("Artifact trained_at must not be empty.")
    if "training_metadata" in bundle and not isinstance(
        bundle["training_metadata"], dict
    ):
        raise TypeError("Artifact training_metadata must be a dictionary.")
    if "library_versions" in bundle and not isinstance(
        bundle["library_versions"], dict
    ):
        raise TypeError("Artifact library_versions must be a dictionary.")

    return bundle
