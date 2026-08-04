"""Reproducible demo CSV and end-to-end semantic contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import re

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from kaggle_bank_fds.scripts.generate_demo_data import (
    COLUMNS,
    build_clean_demo,
    build_exact_overlap_demo,
    build_partial_overlap_demo,
    build_rounded_full_balance_demo,
    generate_demo_files,
)
from kaggle_bank_fds.src.persistence import FdsResultRepository
from kaggle_bank_fds.src.services.fds_analysis_service import FdsAnalysisService
from kaggle_bank_fds.src.ui.csv_preflight import parse_and_validate_paysim_csv


ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
SCRIPT = ROOT / "scripts" / "generate_demo_data.py"
FILENAMES = (
    "clean.csv", "exact_overlap.csv", "partial_overlap.csv",
    "rounded_full_balance.csv",
)
CREATED = datetime(2026, 8, 4, tzinfo=timezone.utc)


def _load_and_analyze(path: Path, tmp_path: Path):
    content = path.read_bytes()
    parsed = parse_and_validate_paysim_csv(
        content, filename=path.name, file_size_bytes=len(content)
    )
    before = parsed.dataframe.copy(deep=True)
    db_path = tmp_path / f"{path.stem}.sqlite3"
    repository = FdsResultRepository.from_path(
        db_path, alert_id_factory=lambda: f"alert-{path.stem}"
    )
    repository.initialize_schema()
    service = FdsAnalysisService(
        repository, source_name=path.name, ruleset_version="bank-fds-demo-v1"
    )
    result = service.analyze_and_persist(
        parsed.dataframe, analysis_run_id=f"run-{path.stem}", created_at=CREATED
    )
    assert_frame_equal(parsed.dataframe, before, check_dtype=True)
    repository.close()
    reopened = FdsResultRepository.from_path(db_path)
    run = reopened.get_analysis_run(f"run-{path.stem}")
    alert = None if result.alert is None else reopened.get_alert_detail(result.alert.summary.alert_id)
    alerts = reopened.list_alerts()
    reopened.close()
    return parsed, result, run, alert, alerts


def _findings(result):
    return {finding.rule_id: finding for finding in result.artifact.rule_findings}


@pytest.mark.parametrize("builder", [
    build_clean_demo, build_exact_overlap_demo, build_partial_overlap_demo,
    build_rounded_full_balance_demo,
])
def test_builders_are_deterministic_and_follow_common_contract(builder):
    first, second = builder(), builder()
    assert_frame_equal(first, second, check_dtype=True)
    assert tuple(first.columns) == COLUMNS
    assert not first.isna().any().any()
    assert len(first) <= 10
    assert all(str(value).startswith("SYNTH_") for value in first["nameOrig"])
    assert all(str(value).startswith("SYNTH_") for value in first["nameDest"])


def test_generation_is_byte_deterministic_and_matches_tracked_files(tmp_path):
    first = generate_demo_files(tmp_path / "first")
    second = generate_demo_files(tmp_path / "second")
    assert tuple(path.name for path in first) == FILENAMES
    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes() == (EXAMPLES / left.name).read_bytes()
        assert left.stat().st_size < 5_000
        frame = pd.read_csv(left, encoding="utf-8")
        assert tuple(frame.columns) == COLUMNS
        assert "Unnamed: 0" not in frame


def test_import_has_no_file_creation_side_effect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = tuple(tmp_path.iterdir())
    spec = importlib.util.spec_from_file_location("isolated_demo_generator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert tuple(tmp_path.iterdir()) == before


def test_tracked_data_is_synthetic_and_contains_no_personal_patterns():
    forbidden = re.compile(r"(?:\b\d{2,3}-\d{3,4}-\d{4}\b|@|\b\d{4}-\d{2}-\d{2}\b)")
    for filename in FILENAMES:
        text = (EXAMPLES / filename).read_text(encoding="utf-8")
        assert not forbidden.search(text)
        assert "SYNTH_" in text


def test_clean_demo_semantics_and_round_trip(tmp_path):
    _, result, run, alert, alerts = _load_and_analyze(EXAMPLES / "clean.csv", tmp_path)
    assert result.prediction["fraud_score"] == result.artifact.fraud_score == run.fraud_score == 0.0
    assert result.prediction["triggered_rules"] == []
    assert len(result.artifact.rule_findings) == 5
    assert sum(len(value.evidence_transaction_ids) for value in result.artifact.rule_findings) == 0
    assert result.alert is alert is None and alerts == ()


def test_exact_overlap_semantics_and_round_trip(tmp_path):
    _, result, run, alert, alerts = _load_and_analyze(EXAMPLES / "exact_overlap.csv", tmp_path)
    findings = _findings(result)
    rapid, split = findings["rapid_repeated_transfer"], findings["split_transaction"]
    assert result.prediction["fraud_score"] == result.artifact.fraud_score == run.fraud_score == .35
    assert result.prediction["triggered_rules"] == ["rapid_repeated_transfer", "split_transaction"]
    assert (rapid.rule_score, split.rule_score) == (30, 35)
    assert rapid.evidence_transaction_ids == split.evidence_transaction_ids
    assert len(rapid.evidence_transaction_ids) == 3
    assert alert is not None and len(alerts) == 1
    persisted = {value.rule_id: value for value in alert.findings}
    assert tuple(value.canonical_transaction_id for value in persisted["rapid_repeated_transfer"].evidence) == rapid.evidence_transaction_ids


def test_partial_overlap_semantics_and_round_trip(tmp_path):
    _, result, run, alert, alerts = _load_and_analyze(EXAMPLES / "partial_overlap.csv", tmp_path)
    findings = _findings(result)
    rapid, split = findings["rapid_repeated_transfer"], findings["split_transaction"]
    assert result.prediction["fraud_score"] == result.artifact.fraud_score == run.fraud_score == .65
    assert (rapid.rule_score, split.rule_score) == (30, 35)
    assert len(rapid.evidence_transaction_ids) == 3
    assert len(split.evidence_transaction_ids) == 4
    assert set(rapid.evidence_transaction_ids) != set(split.evidence_transaction_ids)
    assert alert is not None and len(alerts) == 1


def test_rounded_full_balance_semantics_and_round_trip(tmp_path):
    _, result, run, alert, alerts = _load_and_analyze(
        EXAMPLES / "rounded_full_balance.csv", tmp_path
    )
    findings = _findings(result)
    assert result.prediction["fraud_score"] == result.artifact.fraud_score == run.fraud_score == .40
    assert result.prediction["triggered_rules"] == ["full_balance_transfer", "rounded_amount"]
    assert findings["full_balance_transfer"].rule_score == 20
    assert findings["rounded_amount"].rule_score == 20
    assert findings["full_balance_transfer"].evidence_transaction_ids == findings["rounded_amount"].evidence_transaction_ids
    assert alert is not None and len(alerts) == 1
