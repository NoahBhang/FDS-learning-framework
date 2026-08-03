"""Contract tests for the explicit Legacy/Plugin shadow comparator."""

from dataclasses import FrozenInstanceError, fields
import inspect

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.models import predictor_shadow_comparison as shadow
from kaggle_bank_fds.src.models.predictor_shadow_comparison import (
    ExecutionFailure,
    KnownDifference,
    RuleComparison,
    ShadowComparisonReport,
    compare_legacy_and_plugin,
)


def _row(action="PAYMENT", amount=100.0, actor="A", target="M", balance=2000.0, step=1):
    return {
        "step": step,
        "type": action,
        "amount": amount,
        "nameOrig": actor,
        "oldbalanceOrg": balance,
        "newbalanceOrig": balance - amount,
        "nameDest": target,
        "oldbalanceDest": 0.0,
        "newbalanceDest": amount,
    }


def _normal(index=None):
    return pd.DataFrame([_row()], index=index)


def _full(index=None):
    return pd.DataFrame([_row("TRANSFER", 1000, "A", "B", 1000)], index=index)


def _pair(index=None, amount=1000.0, full=False):
    balance = amount if full else 2000.0
    return pd.DataFrame([
        _row("TRANSFER", amount, "SOURCE", "MIDDLE", balance, 1),
        _row("CASH_OUT", amount, "MIDDLE", "SINK", 2000.0, 2),
    ], index=index)


def _detail(triggered=False, score=0, risk="risk", reason="reason", evidence=None):
    return {"risk_type": risk, "is_suspicious": triggered, "risk_score": score,
            "reason": reason, "evidence_ids": [] if evidence is None else evidence}


def _result(*, score=0.0, triggered=None, details=None, skipped=None):
    return {"fraud_score": score, "triggered_rules": [] if triggered is None else triggered,
            "details": {**({} if details is None else details),
                        "skipped_rules": {} if skipped is None else skipped}}


def test_public_import_and_signature():
    assert list(inspect.signature(compare_legacy_and_plugin).parameters) == ["transaction_data"]


@pytest.mark.parametrize("value", [None, [], {}, "frame"])
def test_non_dataframe_rejected(value):
    with pytest.raises(TypeError, match="DataFrame"):
        compare_legacy_and_plugin(value)


def test_empty_rejected():
    with pytest.raises(ValueError, match="empty"):
        compare_legacy_and_plugin(pd.DataFrame())


@pytest.mark.parametrize("model", [ExecutionFailure, RuleComparison, ShadowComparisonReport])
def test_models_are_frozen_and_slotted(model):
    assert "__slots__" in model.__dict__
    assert model.__dataclass_params__.frozen


def test_report_uses_tuple_snapshots_and_contains_no_raw_payload_fields():
    report = compare_legacy_and_plugin(_normal())
    assert isinstance(report.rule_comparisons, tuple)
    assert isinstance(report.known_differences, tuple)
    assert isinstance(report.unexpected_differences, tuple)
    assert not {"transaction_data", "legacy_result", "plugin_result"} & {f.name for f in fields(report)}
    with pytest.raises(FrozenInstanceError):
        report.equivalent = False


def test_repr_and_equality_are_deterministic():
    assert compare_legacy_and_plugin(_normal()) == compare_legacy_and_plugin(_normal())
    assert repr(compare_legacy_and_plugin(_normal())) == repr(compare_legacy_and_plugin(_normal()))


@pytest.mark.parametrize("frame", [
    _normal(), _full(), _pair(), _pair(full=True),
    _pair(index=["transfer", "cashout"]), _pair(index=[10, 30]),
])
def test_real_predictors_report_expected_default_rule_expansion(frame):
    report = compare_legacy_and_plugin(frame)
    assert report.strict_equivalent is False
    assert report.equivalent is False
    assert report.total_score_matches is True
    assert report.triggered_members_match is True
    assert report.triggered_order_matches is True
    assert set(report.unexpected_differences) == {
        f"{difference}:{rule_id}"
        for rule_id in (
            "rounded_amount",
            "rapid_repeated_transfer",
            "split_transaction",
        )
        for difference in (
            "state_mismatch",
            "triggered_mismatch",
            "score_mismatch",
            "risk_type_mismatch",
        )
    }


def test_duplicate_index_is_snapshotted_without_hashing():
    report = compare_legacy_and_plugin(_pair(index=["same", "same"]))
    comparison = report.rule_comparisons[0]
    assert comparison.legacy_evidence_ids == ("same",)
    assert comparison.evidence_matches


def test_reason_difference_does_not_affect_equivalence(monkeypatch):
    base = _result(details={"transfer_cash_out": _detail(reason="legacy")})
    other = _result(details={"transfer_cash_out": _detail(reason="plugin")})
    monkeypatch.setattr(shadow, "predict", lambda frame: base)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: other)
    report = compare_legacy_and_plugin(_normal())
    assert report.strict_equivalent and report.equivalent
    assert report.rule_comparisons[0].differences == ("reason_mismatch",)


@pytest.mark.parametrize(("change", "code"), [
    ({"triggered": True}, "triggered_mismatch:transfer_cash_out"),
    ({"score": 20}, "score_mismatch:transfer_cash_out"),
    ({"risk": "other"}, "risk_type_mismatch:transfer_cash_out"),
    ({"evidence": [2]}, "evidence_mismatch:transfer_cash_out"),
])
def test_rule_mismatches_are_unexpected(monkeypatch, change, code):
    left = _result(details={"transfer_cash_out": _detail()})
    right = _result(details={"transfer_cash_out": _detail(**change)})
    monkeypatch.setattr(shadow, "predict", lambda frame: left)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: right)
    report = compare_legacy_and_plugin(_normal())
    assert code in report.unexpected_differences
    assert not report.equivalent


def test_legacy_only_and_plugin_only_rules_are_visible(monkeypatch):
    left = _result(details={"legacy_extra": _detail()})
    right = _result(details={"plugin_extra": _detail()})
    monkeypatch.setattr(shadow, "predict", lambda frame: left)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: right)
    report = compare_legacy_and_plugin(_normal())
    assert [item.rule_id for item in report.rule_comparisons] == [
        "transfer_cash_out", "full_balance_transfer", "legacy_extra", "plugin_extra"]
    assert not report.equivalent


def test_triggered_member_and_order_are_independent(monkeypatch):
    details = {"a": _detail(True, 10), "b": _detail(True, 10)}
    monkeypatch.setattr(shadow, "predict", lambda frame: _result(score=.0, triggered=["a", "b"], details=details))
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: _result(score=.0, triggered=["b", "a"], details=details))
    report = compare_legacy_and_plugin(_normal())
    assert report.triggered_members_match
    assert not report.triggered_order_matches
    assert "triggered_order_mismatch" in report.unexpected_differences


def test_skipped_message_is_redacted(monkeypatch):
    monkeypatch.setattr(shadow, "predict", lambda frame: _result(skipped={"x": "secret one"}))
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: _result(skipped={"x": "secret two"}))
    report = compare_legacy_and_plugin(_normal())
    rendered = repr(report)
    assert "secret" not in rendered
    assert "skipped_message_mismatch:x" in report.unexpected_differences


def test_zero_amount_pair_is_narrow_known_difference():
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert not report.strict_equivalent
    assert not report.equivalent
    assert report.known_differences == (KnownDifference.ZERO_AMOUNT_TRANSFER_CASH_OUT,)
    assert "total_score_mismatch" in report.unexpected_differences
    assert {
        "state_mismatch:rounded_amount",
        "state_mismatch:rapid_repeated_transfer",
        "state_mismatch:split_transaction",
    } <= set(report.unexpected_differences)


def test_zero_amount_does_not_hide_unrelated_mismatch(monkeypatch):
    left = _result(score=.25, triggered=["transfer_cash_out"], details={
        "transfer_cash_out": _detail(True, 25, evidence=[0, 1]), "other": _detail()})
    right = _result(details={"transfer_cash_out": _detail(), "other": _detail(risk="changed")})
    monkeypatch.setattr(shadow, "predict", lambda frame: left)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: right)
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert KnownDifference.ZERO_AMOUNT_TRANSFER_CASH_OUT in report.known_differences
    assert "risk_type_mismatch:other" in report.unexpected_differences
    assert not report.equivalent


@pytest.mark.parametrize(("target", "stage"), [
    ("predict", "legacy_predictor"), ("predict_with_plugins", "plugin_predictor")])
def test_execution_failure_is_redacted(monkeypatch, target, stage):
    def fail(frame):
        raise RuntimeError("account SECRET amount 999 traceback")
    monkeypatch.setattr(shadow, target, fail)
    report = compare_legacy_and_plugin(_normal())
    failure = report.legacy_execution_failure or report.plugin_execution_failure
    assert failure == ExecutionFailure(stage, "RuntimeError")
    assert "SECRET" not in repr(report)
    assert not report.equivalent


def test_both_execution_failures_are_preserved(monkeypatch):
    def fail(frame):
        raise RuntimeError("hidden")
    monkeypatch.setattr(shadow, "predict", fail)
    monkeypatch.setattr(shadow, "predict_with_plugins", fail)
    report = compare_legacy_and_plugin(_normal())
    assert report.legacy_execution_failure and report.plugin_execution_failure


@pytest.mark.parametrize("bad", [
    {}, {"fraud_score": True, "triggered_rules": [], "details": {"skipped_rules": {}}},
    {"fraud_score": np.inf, "triggered_rules": [], "details": {"skipped_rules": {}}},
    {"fraud_score": 0.0, "triggered_rules": "x", "details": {"skipped_rules": {}}},
])
def test_invalid_success_schema_propagates(monkeypatch, bad):
    monkeypatch.setattr(shadow, "predict", lambda frame: bad)
    with pytest.raises((TypeError, ValueError)):
        compare_legacy_and_plugin(_normal())


@pytest.mark.parametrize("target", ["predict", "predict_with_plugins"])
def test_predictor_input_mutation_invariant_propagates(monkeypatch, target):
    def mutate(frame):
        frame.loc[frame.index[0], "amount"] = 999
        return _result()
    monkeypatch.setattr(shadow, target, mutate)
    with pytest.raises(AssertionError):
        compare_legacy_and_plugin(_normal())


def test_original_input_is_unchanged():
    frame = _pair(index=["t", "c"])
    snapshot = frame.copy(deep=True)
    compare_legacy_and_plugin(frame)
    assert_frame_equal(frame, snapshot)


@pytest.mark.parametrize(("left", "right", "expected"), [
    ([np.nan], [np.nan], True), ([pd.NA], [pd.NA], True),
    ([pd.Timestamp("2026-01-01")], [pd.Timestamp("2026-01-01")], True),
    ([("a", 1)], [("a", 1)], True), ([[1, 2]], [[1, 2]], True),
    ([[1, 2]], [[1, 3]], False), ([1], [1, 2], False),
])
def test_evidence_safe_positional_equality(monkeypatch, left, right, expected):
    monkeypatch.setattr(shadow, "predict", lambda frame: _result(details={"x": _detail(evidence=left)}))
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: _result(details={"x": _detail(evidence=right)}))
    comparison = compare_legacy_and_plugin(_normal()).rule_comparisons[-1]
    assert comparison.evidence_matches is expected


def test_no_stdout_or_stderr(capsys):
    compare_legacy_and_plugin(_normal())
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def _patch_transfer_mismatch(monkeypatch, *, unrelated=False):
    legacy_details = {
        "transfer_cash_out": _detail(True, 25, evidence=[0, 1]),
    }
    plugin_details = {"transfer_cash_out": _detail()}
    if unrelated:
        legacy_details["other"] = _detail(risk="legacy")
        plugin_details["other"] = _detail(risk="plugin")
    monkeypatch.setattr(
        shadow,
        "predict",
        lambda frame: _result(
            score=.25,
            triggered=["transfer_cash_out"],
            details=legacy_details,
        ),
    )
    monkeypatch.setattr(
        shadow,
        "predict_with_plugins",
        lambda frame: _result(details=plugin_details),
    )


def test_unlinked_zero_amount_rows_are_not_known(monkeypatch):
    _patch_transfer_mismatch(monkeypatch)
    frame = _pair(amount=0.0)
    frame.loc[frame.index[1], "nameOrig"] = "UNRELATED"
    report = compare_legacy_and_plugin(frame)
    assert report.known_differences == ()
    assert report.unexpected_differences
    assert not report.equivalent


@pytest.mark.parametrize(("transfer_step", "cashout_step"), [(1, 26), (2, 1)])
def test_invalid_zero_amount_step_gap_is_not_known(
    monkeypatch, transfer_step, cashout_step
):
    _patch_transfer_mismatch(monkeypatch)
    frame = _pair(amount=0.0)
    frame["step"] = [transfer_step, cashout_step]
    report = compare_legacy_and_plugin(frame)
    assert report.known_differences == ()
    assert not report.equivalent


def test_connected_zero_pair_with_unrelated_mismatch_is_not_equivalent(monkeypatch):
    _patch_transfer_mismatch(monkeypatch, unrelated=True)
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert report.known_differences == (
        KnownDifference.ZERO_AMOUNT_TRANSFER_CASH_OUT,
    )
    assert "risk_type_mismatch:other" in report.unexpected_differences
    assert not report.equivalent


def test_connected_zero_pair_does_not_hide_unrelated_trigger_order(monkeypatch):
    legacy_details = {
        "transfer_cash_out": _detail(True, 25),
        "a": _detail(True, 10),
        "b": _detail(True, 10),
    }
    plugin_details = {
        "transfer_cash_out": _detail(),
        "a": _detail(True, 10),
        "b": _detail(True, 10),
    }
    monkeypatch.setattr(
        shadow,
        "predict",
        lambda frame: _result(
            score=.45,
            triggered=["transfer_cash_out", "a", "b"],
            details=legacy_details,
        ),
    )
    monkeypatch.setattr(
        shadow,
        "predict_with_plugins",
        lambda frame: _result(
            score=.20, triggered=["b", "a"], details=plugin_details
        ),
    )
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert "triggered_order_mismatch" in report.unexpected_differences
    assert not report.equivalent


def test_connected_zero_pair_does_not_hide_unexplained_total_delta(monkeypatch):
    _patch_transfer_mismatch(monkeypatch)
    original = shadow.predict
    monkeypatch.setattr(
        shadow,
        "predict",
        lambda frame: {**original(frame), "fraud_score": .30},
    )
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert "total_score_mismatch" in report.unexpected_differences
    assert not report.equivalent


def test_total_score_only_mismatch_is_unexpected(monkeypatch):
    details = {"transfer_cash_out": _detail()}
    monkeypatch.setattr(
        shadow, "predict", lambda frame: _result(score=.1, details=details)
    )
    monkeypatch.setattr(
        shadow, "predict_with_plugins", lambda frame: _result(details=details)
    )
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert not report.total_score_matches
    assert report.known_differences == ()
    assert report.unexpected_differences == ("total_score_mismatch",)
    assert not report.strict_equivalent and not report.equivalent


def test_triggered_duplicate_count_mismatch(monkeypatch):
    detail = {"transfer_cash_out": _detail(True, 25)}
    monkeypatch.setattr(
        shadow,
        "predict",
        lambda frame: _result(
            score=.25,
            triggered=["transfer_cash_out", "transfer_cash_out"],
            details=detail,
        ),
    )
    monkeypatch.setattr(
        shadow,
        "predict_with_plugins",
        lambda frame: _result(
            score=.25, triggered=["transfer_cash_out"], details=detail
        ),
    )
    report = compare_legacy_and_plugin(_normal())
    assert not report.triggered_members_match
    assert not report.triggered_order_matches
    assert "triggered_members_mismatch" in report.unexpected_differences
    assert not report.equivalent


@pytest.mark.parametrize(("left", "right", "expected"), [
    (np.nan, np.array([np.nan]), False),
    (np.array([np.nan]), np.array([[np.nan]]), False),
    (np.array([np.nan]), np.array([np.nan]), True),
    (np.array([1.0, np.nan]), np.array([1.0, np.nan]), True),
    (np.array([1.0, np.nan]), np.array([np.nan, 1.0]), False),
    (pd.NA, pd.NA, True),
])
def test_safe_equality_distinguishes_scalar_shape_and_order(left, right, expected):
    assert shadow._safe_equal(left, right) is expected


class _AmbiguousEquality:
    def __eq__(self, other):
        return np.array([True, True])


class _ExplodingEquality:
    def __eq__(self, other):
        raise RuntimeError("must be treated as unequal")


@pytest.mark.parametrize("value", [_AmbiguousEquality(), _ExplodingEquality()])
def test_ambiguous_or_raising_object_equality_is_false(value):
    assert shadow._safe_equal(value, object()) is False


def _snapshot_report(monkeypatch, evidence):
    result = _result(details={"x": _detail(evidence=evidence)})
    monkeypatch.setattr(shadow, "predict", lambda frame: result)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: result)
    return compare_legacy_and_plugin(_normal())


@pytest.mark.parametrize("evidence,mutate", [
    ([[1, 2]], lambda value: value[0].append(3)),
    ([([1, 2],)], lambda value: value[0][0].append(3)),
])
def test_nested_mutable_evidence_is_snapshotted(monkeypatch, evidence, mutate):
    report = _snapshot_report(monkeypatch, evidence)
    before_repr = repr(report)
    before_equality = report == report
    mutate(evidence)
    assert repr(report) == before_repr
    assert report == report
    assert before_equality
    assert report.rule_comparisons[-1].legacy_evidence_ids[0] is not evidence[0]


def test_numpy_evidence_snapshot_preserves_dtype_shape_and_value(monkeypatch):
    array = np.array([[1.0, np.nan]])
    report = _snapshot_report(monkeypatch, [array])
    before = repr(report)
    array[0, 0] = 99.0
    assert repr(report) == before
    snapshot = report.rule_comparisons[-1].legacy_evidence_ids[0]
    assert snapshot.shape == (1, 2)
    assert snapshot.dtype == np.dtype(float).str
    assert snapshot is not array


def test_array_snapshot_distinguishes_shape_and_has_stable_equality(monkeypatch):
    left = _result(details={"x": _detail(evidence=[np.array([np.nan])])})
    right = _result(details={"x": _detail(evidence=[np.array([[np.nan]])])})
    monkeypatch.setattr(shadow, "predict", lambda frame: left)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: right)
    report = compare_legacy_and_plugin(_normal())
    assert not report.rule_comparisons[-1].evidence_matches
    assert report == report
    assert repr(report) == repr(report)


def test_unsupported_mutable_evidence_type_is_rejected(monkeypatch):
    class MutableEvidence:
        pass

    with pytest.raises(TypeError, match="unsupported mutable"):
        _snapshot_report(monkeypatch, [MutableEvidence()])


@pytest.mark.parametrize(
    ("legacy_total", "plugin_total", "legacy_rule_score", "shared_score"),
    [(.55, .30, 25, 30), (.30, .10, 20, 10)],
)
def test_known_score_delta_uses_integer_points(
    monkeypatch,
    legacy_total,
    plugin_total,
    legacy_rule_score,
    shared_score,
):
    shared = _detail(True, shared_score)
    legacy = _result(
        score=legacy_total,
        triggered=["transfer_cash_out", "shared"],
        details={
            "transfer_cash_out": _detail(True, legacy_rule_score, evidence=[0, 1]),
            "shared": shared,
        },
    )
    plugin = _result(
        score=plugin_total,
        triggered=["shared"],
        details={"transfer_cash_out": _detail(), "shared": shared},
    )
    monkeypatch.setattr(shadow, "predict", lambda frame: legacy)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: plugin)
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert not report.total_score_matches
    assert "total_score_mismatch" not in report.unexpected_differences
    assert report.equivalent


def test_known_score_delta_does_not_explain_different_point_delta(monkeypatch):
    legacy = _result(
        score=.35,
        triggered=["transfer_cash_out"],
        details={"transfer_cash_out": _detail(True, 20, evidence=[0, 1])},
    )
    plugin = _result(details={"transfer_cash_out": _detail()})
    monkeypatch.setattr(shadow, "predict", lambda frame: legacy)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: plugin)
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert "total_score_mismatch" in report.unexpected_differences
    assert not report.equivalent


def test_other_rule_score_mismatch_prevents_total_attribution(monkeypatch):
    legacy = _result(
        score=.55,
        triggered=["transfer_cash_out"],
        details={
            "transfer_cash_out": _detail(True, 25, evidence=[0, 1]),
            "other": _detail(score=10),
        },
    )
    plugin = _result(
        score=.30,
        details={"transfer_cash_out": _detail(), "other": _detail(score=5)},
    )
    monkeypatch.setattr(shadow, "predict", lambda frame: legacy)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: plugin)
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert "total_score_mismatch" in report.unexpected_differences
    assert "score_mismatch:other" in report.unexpected_differences
    assert not report.equivalent


def test_score_cap_keeps_total_match_and_rule_difference(monkeypatch):
    shared = _detail(True, 100)
    legacy = _result(
        score=1.0,
        triggered=["transfer_cash_out", "shared"],
        details={
            "transfer_cash_out": _detail(True, 25, evidence=[0, 1]),
            "shared": shared,
        },
    )
    plugin = _result(
        score=1.0,
        triggered=["shared"],
        details={"transfer_cash_out": _detail(), "shared": shared},
    )
    monkeypatch.setattr(shadow, "predict", lambda frame: legacy)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: plugin)
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert report.total_score_matches
    assert not report.rule_comparisons[0].score_matches
    assert not report.strict_equivalent and report.equivalent


def _patch_duplicate_trigger_results(monkeypatch, legacy_rules, plugin_rules):
    legacy = _result(
        score=.25,
        triggered=legacy_rules,
        details={"transfer_cash_out": _detail(True, 25, evidence=[0, 1])},
    )
    plugin = _result(
        triggered=plugin_rules,
        details={"transfer_cash_out": _detail()},
    )
    monkeypatch.setattr(shadow, "predict", lambda frame: legacy)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: plugin)


def test_one_extra_legacy_transfer_occurrence_is_attributed(monkeypatch):
    _patch_duplicate_trigger_results(
        monkeypatch,
        ["transfer_cash_out", "transfer_cash_out"],
        ["transfer_cash_out"],
    )
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert not report.triggered_members_match
    assert not report.triggered_order_matches
    assert "triggered_members_mismatch" not in report.unexpected_differences
    assert "triggered_order_mismatch" not in report.unexpected_differences
    assert report.equivalent


def test_two_extra_legacy_transfer_occurrences_are_not_attributed(monkeypatch):
    _patch_duplicate_trigger_results(
        monkeypatch,
        ["transfer_cash_out", "transfer_cash_out"],
        [],
    )
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert "triggered_members_mismatch" in report.unexpected_differences
    assert "triggered_order_mismatch" in report.unexpected_differences
    assert not report.equivalent


def test_extra_plugin_transfer_occurrence_is_not_attributed(monkeypatch):
    _patch_duplicate_trigger_results(
        monkeypatch,
        ["transfer_cash_out"],
        ["transfer_cash_out", "transfer_cash_out"],
    )
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert "triggered_members_mismatch" in report.unexpected_differences
    assert not report.equivalent


def test_duplicate_difference_without_zero_pair_is_not_known(monkeypatch):
    _patch_duplicate_trigger_results(
        monkeypatch,
        ["transfer_cash_out", "transfer_cash_out"],
        ["transfer_cash_out"],
    )
    report = compare_legacy_and_plugin(_normal())
    assert report.known_differences == ()
    assert not report.equivalent


def test_one_occurrence_removal_preserves_unrelated_member_mismatch(monkeypatch):
    legacy = _result(
        score=.25,
        triggered=["transfer_cash_out", "full_balance_transfer"],
        details={
            "transfer_cash_out": _detail(True, 25, evidence=[0, 1]),
            "full_balance_transfer": _detail(),
        },
    )
    plugin = _result(
        triggered=["other"],
        details={"transfer_cash_out": _detail(), "other": _detail()},
    )
    monkeypatch.setattr(shadow, "predict", lambda frame: legacy)
    monkeypatch.setattr(shadow, "predict_with_plugins", lambda frame: plugin)
    report = compare_legacy_and_plugin(_pair(amount=0.0))
    assert "triggered_members_mismatch" in report.unexpected_differences
    assert not report.equivalent
