"""Contract tests for RoundedAmountRule."""

from datetime import datetime
from decimal import Decimal
import inspect

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.rounded_amount_rule import RoundedAmountRule
from kaggle_bank_fds.src.rules.rule_engine import RuleEngine
from kaggle_bank_fds.src.rules.rule_registry import RuleRegistry
from kaggle_bank_fds.src.rules.rule_result import RuleResult


REQUIRED = RoundedAmountRule.REQUIRED_COLUMNS


def _frame(*changes, index=None):
    rows = []
    for position, change in enumerate(changes or ({},)):
        rows.append({
            "transaction_id": f"tx-{position}", "source_row_id": position,
            "step": position + 1, "transaction_datetime": pd.Timestamp("2026-01-01"),
            "action_type": "TRANSFER", "amount": 100_000.0,
            "actor_account": "A", "target_account": "B",
        } | change)
    return pd.DataFrame(rows, columns=REQUIRED, index=index)


def test_metadata_signature_defaults_and_properties():
    rule = RoundedAmountRule()
    assert (rule.rule_id, rule.rule_name) == ("rounded_amount", "Rounded Amount Rule")
    assert list(inspect.signature(RoundedAmountRule).parameters) == ["min_amount", "rounding_unit", "score"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in inspect.signature(RoundedAmountRule).parameters.values())
    assert (rule.min_amount, rule.rounding_unit, rule.score) == (100_000.0, 10_000, 20)
    with pytest.raises(AttributeError): rule.score = 1


def test_numpy_parameters_normalize_to_python_types():
    rule = RoundedAmountRule(min_amount=np.float64(20), rounding_unit=np.int64(10), score=np.int64(7))
    assert type(rule.min_amount) is float
    assert type(rule.rounding_unit) is type(rule.score) is int


@pytest.mark.parametrize(("name", "value"), [
    ("min_amount", True), ("min_amount", "100"), ("min_amount", Decimal("100")),
    ("min_amount", 1+0j), ("min_amount", None),
    ("rounding_unit", True), ("rounding_unit", "10"), ("rounding_unit", Decimal("10")),
    ("rounding_unit", 1+0j), ("rounding_unit", None),
    ("score", True), ("score", "20"), ("score", Decimal("20")),
    ("score", 20.0), ("score", None),
])
def test_invalid_parameter_types(name, value):
    with pytest.raises(TypeError): RoundedAmountRule(**{name: value})


@pytest.mark.parametrize(("kwargs"), [
    {"min_amount": 0}, {"min_amount": np.nan}, {"min_amount": np.inf},
    {"rounding_unit": 0}, {"rounding_unit": -1}, {"score": 0}, {"score": 101},
    {"min_amount": 9, "rounding_unit": 10},
])
def test_invalid_parameter_values(kwargs):
    with pytest.raises(ValueError): RoundedAmountRule(**kwargs)


@pytest.mark.parametrize(("amount", "detected"), [
    (90_000.0, False), (100_000.0, True), (110_000.0, True),
    (105_000.0, False), (100_000.5, False), (0.0, False), (-100_000.0, False),
])
def test_detection_boundaries(amount, detected):
    result = RoundedAmountRule().evaluate(_frame({"amount": amount}))
    assert result.triggered is detected
    assert result.score == (20 if detected else 0)
    assert isinstance(result.evidence, tuple)


@pytest.mark.parametrize("action", ["TRANSFER", "CASH_OUT"])
def test_both_candidate_actions_detect(action):
    assert RoundedAmountRule().evaluate(_frame({"action_type": action})).triggered


def test_non_candidate_malformed_values_are_ignored():
    bad = {"action_type":"PAYMENT", "transaction_id":"", "source_row_id":True,
           "step":"bad", "amount":"bad", "actor_account":None,
           "target_account":None, "transaction_datetime":"bad"}
    result = RoundedAmountRule().evaluate(_frame(bad))
    assert not result.triggered


@pytest.mark.parametrize(("field", "value", "error"), [
    ("amount", pd.NA, ValueError), ("amount", np.inf, ValueError),
    ("amount", True, TypeError), ("amount", "100000", TypeError),
    ("amount", 1+0j, TypeError), ("step", pd.NA, ValueError),
    ("step", True, TypeError), ("step", 1.0, TypeError),
    ("actor_account", None, ValueError), ("target_account", " ", ValueError),
    ("transaction_id", pd.NA, ValueError), ("transaction_id", 1, TypeError),
])
def test_invalid_candidate_fields(field, value, error):
    with pytest.raises(error): RoundedAmountRule().evaluate(_frame({field:value}))


def test_duplicate_transaction_rejected_but_duplicate_source_allowed():
    with pytest.raises(ValueError, match="unique"):
        RoundedAmountRule().evaluate(_frame({"transaction_id":"x"},{"transaction_id":"x"}))
    result = RoundedAmountRule().evaluate(_frame({"source_row_id":7},{"source_row_id":7}))
    assert [item.source_row_id for item in result.evidence] == [7, 7]


def test_typed_empty_clean_and_untyped_empty_fails():
    result = RoundedAmountRule().evaluate(_frame().iloc[:0])
    assert (result.triggered, result.score, result.evidence) == (False, 0, ())
    with pytest.raises(ValueError, match="Missing"): RoundedAmountRule().evaluate(pd.DataFrame())


@pytest.mark.parametrize("missing", REQUIRED)
def test_each_required_column_is_enforced(missing):
    with pytest.raises(ValueError, match=missing): RoundedAmountRule().evaluate(_frame().drop(columns=missing))


@pytest.mark.parametrize("index", [[20,1], ["z","a"], ["same","same"]])
def test_evidence_uses_positional_order_not_index(index):
    frame = _frame({"transaction_id":"first"},{"transaction_id":"second"}, index=index)
    result = RoundedAmountRule().evaluate(frame)
    assert [item.transaction_id for item in result.evidence] == ["first", "second"]
    assert result.evidence[0].transaction_datetime == datetime(2026,1,1)
    assert "A" not in result.reason and "B" not in result.evidence[0].message


def test_input_immutable_and_repeatable():
    frame = _frame({}, {}, index=["b","a"]); before = frame.copy(deep=True)
    first = RoundedAmountRule().evaluate(frame); second = RoundedAmountRule().evaluate(frame)
    assert first == second; assert_frame_equal(frame, before)


def test_non_dataframe_source_id_datetime_and_complete_evidence():
    with pytest.raises(TypeError, match="DataFrame"): RoundedAmountRule().evaluate([])
    with pytest.raises(TypeError, match="source_row_id"):
        RoundedAmountRule().evaluate(_frame({"source_row_id":True}))
    with pytest.raises(TypeError, match="datetime"):
        RoundedAmountRule().evaluate(_frame({"transaction_datetime":"bad"}))
    result=RoundedAmountRule().evaluate(_frame({"transaction_id":" TX ","source_row_id":pd.NA,"actor_account":"actor","target_account":"target","amount":100_000}))
    item=result.evidence[0]
    assert (item.transaction_id,item.source_row_id,item.actor_account,item.target_account,item.amount)==("TX",None,"actor","target",100_000.0)


class _FailRule(BaseRule):
    rule_id="fail"; rule_name="Fail"; description="Fails."
    def evaluate(self, transactions): raise RuntimeError("boom")


def test_registry_engine_and_failure_isolation():
    registry=RuleRegistry(); registry.register(_FailRule()); registry.register(RoundedAmountRule())
    report=RuleEngine(registry).evaluate(_frame())
    assert [r.rule_id for r in report.results] == ["rounded_amount"]
    assert [e.rule_id for e in report.errors] == ["fail"]
    assert isinstance(report.results[0], RuleResult)


def _object_amount_frame(value, *, action_type="TRANSFER"):
    frame = _frame({"action_type": action_type})
    frame["amount"] = pd.Series([value], dtype=object, index=frame.index)
    return frame


def test_exact_large_python_integral_divisibility_and_evidence():
    value = 10**100
    frame = _object_amount_frame(value)
    before = frame.copy(deep=True)
    result = RoundedAmountRule().evaluate(frame)
    assert result.triggered
    assert len(result.evidence) == 1
    assert result.evidence[0].amount == float(value)
    assert np.isfinite(result.evidence[0].amount)
    assert_frame_equal(frame, before)


def test_exact_large_python_integral_nondivisible_is_clean():
    result = RoundedAmountRule().evaluate(_object_amount_frame(10**100 + 1))
    assert not result.triggered
    assert result.evidence == ()


def test_exact_large_negative_python_integral_is_clean():
    result = RoundedAmountRule().evaluate(_object_amount_frame(-(10**100)))
    assert not result.triggered


def test_float_range_overflow_becomes_redacted_value_error():
    value = 10**400
    with pytest.raises(ValueError, match="supported finite float range") as error:
        RoundedAmountRule().evaluate(_object_amount_frame(value))
    assert type(error.value) is ValueError
    assert str(value) not in str(error.value)


def test_large_numpy_integral_uses_exact_modulo():
    unit = 1_000
    value = np.int64(9_000_000_000_000_000_000)
    result = RoundedAmountRule(
        min_amount=unit,
        rounding_unit=unit,
    ).evaluate(_object_amount_frame(value))
    assert result.triggered
    assert result.evidence[0].amount == float(value)


def test_large_float_uses_stored_float_value():
    result = RoundedAmountRule().evaluate(_object_amount_frame(float(1e20)))
    assert result.triggered
    assert result.evidence[0].amount == 1e20


def test_non_candidate_giant_integral_is_ignored():
    result = RoundedAmountRule().evaluate(
        _object_amount_frame(10**400, action_type="PAYMENT")
    )
    assert not result.triggered
    assert result.evidence == ()
