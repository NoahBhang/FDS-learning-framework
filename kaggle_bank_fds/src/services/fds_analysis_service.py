"""End-to-end orchestration from raw transactions to persisted typed results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import pandas as pd

from kaggle_bank_fds.src.persistence.analysis_artifact_builder import (
    analyze_with_plugins_for_persistence,
)
from kaggle_bank_fds.src.persistence.fds_result_repository import FdsResultRepository
from kaggle_bank_fds.src.persistence.persistence_models import AnalysisPersistenceArtifact
from kaggle_bank_fds.src.persistence.persistence_read_models import (
    AlertDetail,
    AnalysisRunRecord,
)
from kaggle_bank_fds.src.rules.base_rule import BaseRule


class AnalysisServiceIntegrityError(RuntimeError):
    """Raised when saved state cannot be assembled into a consistent response."""


@dataclass(frozen=True, slots=True)
class AnalysisServiceResult:
    prediction: dict
    artifact: AnalysisPersistenceArtifact
    analysis_run: AnalysisRunRecord
    alert: AlertDetail | None


class FdsAnalysisService:
    """Analyze raw transactions once, persist the artifact, and read it back."""

    def __init__(
        self,
        repository: FdsResultRepository,
        *,
        source_name: str,
        ruleset_version: str,
    ) -> None:
        if not isinstance(repository, FdsResultRepository):
            raise TypeError("repository must be FdsResultRepository.")
        self._repository = repository
        self._source_name = _required_text(source_name, "source_name")
        self._ruleset_version = _required_text(ruleset_version, "ruleset_version")

    def analyze_and_persist(
        self,
        raw_data: pd.DataFrame,
        *,
        rules: Sequence[BaseRule] | None = None,
        analysis_run_id: str | None = None,
        created_at: datetime | None = None,
    ) -> AnalysisServiceResult:
        active_rules = None if rules is None else list(rules)
        prediction, artifact = analyze_with_plugins_for_persistence(
            raw_data,
            source_name=self._source_name,
            ruleset_version=self._ruleset_version,
            rules=active_rules,
            analysis_run_id=analysis_run_id,
            created_at=created_at,
        )
        if prediction.get("fraud_score") != artifact.fraud_score:
            raise AnalysisServiceIntegrityError(
                "Prediction and persistence artifact scores differ."
            )
        stored_run_id = self._repository.save_analysis(artifact)
        analysis_run = self._repository.get_analysis_run(stored_run_id)
        if analysis_run is None:
            raise AnalysisServiceIntegrityError("Saved analysis run could not be read back.")

        alert_expected = artifact.fraud_score > 0.0 or any(
            finding.triggered for finding in artifact.rule_findings
        )
        alert_summary = self._repository.get_alert_by_run_id(stored_run_id)
        if alert_expected and alert_summary is None:
            raise AnalysisServiceIntegrityError("Expected alert could not be read back.")
        if not alert_expected and alert_summary is not None:
            raise AnalysisServiceIntegrityError("Clean analysis unexpectedly produced an alert.")
        alert = None
        if alert_summary is not None:
            alert = self._repository.get_alert_detail(alert_summary.alert_id)
            if alert is None:
                raise AnalysisServiceIntegrityError("Alert detail could not be read back.")
        return AnalysisServiceResult(
            prediction=deepcopy(prediction),
            artifact=artifact,
            analysis_run=analysis_run,
            alert=alert,
        )


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized
