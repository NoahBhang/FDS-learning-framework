"""Plugin FullBalanceTransferRule 계약 테스트."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.full_balance_transfer_rule import (
    FullBalanceTransferRule,
)
from kaggle_bank_fds.src.rules.rule_engine import RuleEngine
from kaggle_bank_fds.src.rules.rule_registry import RuleRegistry
from kaggle_bank_fds.src.rules.rule_result import RuleResult


REQUIRED_COLUMNS = FullBalanceTransferRule.REQUIRED_COLUMNS


def _frame(*rows: dict, index=None) -> pd.DataFrame:
    defaults = {
        "transaction_id": "tx-1",
        "source_row_id": 0,
        "transaction_datetime": pd.Timestamp("2026-08-03 10:00:00"),
        "action_type": "TRANSFER",
        "amount": 999.0,
        "balance_before": 1000.0,
        "actor_account": "actor-1",
        "target_account": "target-1",
    }
    normalized = []
    for position, changes in enumerate(rows or ({},)):
        row = defaults | changes
        if "transaction_id" not in changes:
            row["transaction_id"] = f"tx-{position + 1}"
        if "source_row_id" not in changes:
            row["source_row_id"] = position
        normalized.append(row)
    return pd.DataFrame(normalized, columns=REQUIRED_COLUMNS, index=index)


def test_metadata_matches_contract() -> None:
    rule = FullBalanceTransferRule()
    assert rule.rule_id == "full_balance_transfer"
    assert rule.rule_name == "Full Balance Transfer Rule"
    assert rule.description == (
        "Detects transfer transactions that move at least the configured ratio "
        "of the originating account's pre-transfer balance, indicating a full "
        "or effectively full balance transfer."
    )


def test_rule_registers_and_can_be_retrieved() -> None:
    rule = FullBalanceTransferRule()
    registry = RuleRegistry()
    registry.register(rule)
    assert registry.get(rule.rule_id) is rule


def test_default_ratio_and_score_constants() -> None:
    rule = FullBalanceTransferRule()
    assert rule.minimum_balance_ratio == 0.999
    assert type(rule.minimum_balance_ratio) is float
    assert (rule.BASE_SCORE, rule.SCORE_PER_MATCH, rule.MAX_SCORE) == (15, 5, 30)


def test_ratio_property_is_readonly() -> None:
    rule = FullBalanceTransferRule()
    with pytest.raises(AttributeError):
        rule.minimum_balance_ratio = 0.5


@pytest.mark.parametrize(
    "value", [1, 0.5, 1.0, np.int32(1), np.int64(1), np.float32(0.75), np.float64(0.8)]
)
def test_supported_ratio_types(value: object) -> None:
    rule = FullBalanceTransferRule(minimum_balance_ratio=value)
    assert rule.minimum_balance_ratio == float(value)
    assert type(rule.minimum_balance_ratio) is float


@pytest.mark.parametrize(
    "value", [True, False, np.bool_(True), np.bool_(False), "0.999", None, 1 + 0j, Decimal("0.999")]
)
def test_unsupported_ratio_types(value: object) -> None:
    with pytest.raises(TypeError):
        FullBalanceTransferRule(minimum_balance_ratio=value)


@pytest.mark.parametrize("value", [0, -0.1, 1.0001, np.nan, np.inf, -np.inf])
def test_invalid_ratio_values(value: float) -> None:
    with pytest.raises(ValueError):
        FullBalanceTransferRule(minimum_balance_ratio=value)


def test_single_detection_complete_result() -> None:
    result = FullBalanceTransferRule().evaluate(_frame({"amount": 1000.0}))
    assert result.triggered is True
    assert result.score == 20
    assert isinstance(result.evidence, tuple)
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.transaction_id == "tx-1"
    assert evidence.source_row_id == 0
    assert evidence.actor_account == "actor-1"
    assert evidence.target_account == "target-1"
    assert evidence.transaction_datetime == datetime(2026, 8, 3, 10, 0)
    assert evidence.amount == 1000.0
    assert evidence.message == (
        "Transferred 100.00% of pre-transfer balance 1000.00; "
        "configured minimum is 99.90%."
    )
    assert result.reason == (
        "Detected 1 full or effectively full balance transfer(s) at or above "
        "the configured minimum ratio of 99.90%."
    )


@pytest.mark.parametrize(
    ("amount", "detected"),
    [(999.0, True), (998.999999, False), (1000.0, True), (1001.0, True)],
)
def test_default_ratio_boundaries(amount: float, detected: bool) -> None:
    assert FullBalanceTransferRule().evaluate(_frame({"amount": amount})).triggered is detected


def test_custom_ratio_and_exact_floating_boundary() -> None:
    ratio = 0.1 + 0.2
    boundary = 10.0 * ratio
    rule = FullBalanceTransferRule(minimum_balance_ratio=ratio)
    assert rule.evaluate(_frame({"amount": boundary, "balance_before": 10.0})).triggered
    assert not rule.evaluate(
        _frame({"amount": np.nextafter(boundary, -np.inf), "balance_before": 10.0})
    ).triggered


def test_ratio_one_requires_full_balance_or_more() -> None:
    rule = FullBalanceTransferRule(minimum_balance_ratio=1.0)
    assert not rule.evaluate(_frame({"amount": 999.999})).triggered
    assert rule.evaluate(_frame({"amount": 1000.0})).triggered


@pytest.mark.parametrize(
    "changes",
    [
        {"balance_before": 0.0, "amount": 1.0},
        {"balance_before": -1.0, "amount": 1.0},
        {"balance_before": 1000.0, "amount": 0.0},
        {"balance_before": 1000.0, "amount": -1.0},
        {"action_type": "CASH_OUT", "amount": 1000.0},
    ],
)
def test_valid_non_matches(changes: dict) -> None:
    result = FullBalanceTransferRule().evaluate(_frame(changes))
    assert result.triggered is False
    assert result.score == 0
    assert result.evidence == ()
    assert result.reason == (
        "Detected 0 full or effectively full balance transfer(s) at or above "
        "the configured minimum ratio of 99.90%."
    )


@pytest.mark.parametrize("field", ["amount", "balance_before"])
@pytest.mark.parametrize("value", [pd.NA, np.nan])
def test_missing_candidate_numeric_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        FullBalanceTransferRule().evaluate(_frame({field: value}))


@pytest.mark.parametrize("field", ["amount", "balance_before"])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_nonfinite_candidate_numeric_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        FullBalanceTransferRule().evaluate(_frame({field: value}))


@pytest.mark.parametrize("field", ["amount", "balance_before"])
@pytest.mark.parametrize(
    "value", [True, np.bool_(True), 1 + 0j, "1000", Decimal("1000"), datetime(2026, 1, 1), timedelta(days=1)]
)
def test_invalid_candidate_numeric_type(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=field):
        FullBalanceTransferRule().evaluate(_frame({field: value}))


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
def test_nullable_numeric_dtypes_are_supported(dtype: str) -> None:
    frame = _frame({"amount": 1000, "balance_before": 1000})
    frame["amount"] = frame["amount"].astype(dtype)
    frame["balance_before"] = frame["balance_before"].astype(dtype)
    assert FullBalanceTransferRule().evaluate(frame).triggered


def test_object_dtype_real_values_are_supported() -> None:
    frame = _frame({"amount": np.float64(1000), "balance_before": np.int64(1000)})
    frame["amount"] = frame["amount"].astype(object)
    frame["balance_before"] = frame["balance_before"].astype(object)
    assert FullBalanceTransferRule().evaluate(frame).triggered


@pytest.mark.parametrize("count,score", [(2, 25), (3, 30), (4, 30), (8, 30)])
def test_multiple_match_score_and_cap(count: int, score: int) -> None:
    rows = tuple({"amount": 1000.0, "actor_account": f"actor-{i}"} for i in range(count))
    result = FullBalanceTransferRule().evaluate(_frame(*rows))
    assert result.score == score
    assert [item.transaction_id for item in result.evidence] == [f"tx-{i + 1}" for i in range(count)]
    assert [item.actor_account for item in result.evidence] == [f"actor-{i}" for i in range(count)]


def test_duplicate_candidate_transaction_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        FullBalanceTransferRule().evaluate(
            _frame({"transaction_id": "same"}, {"transaction_id": " same "})
        )


def test_duplicate_source_row_id_is_allowed() -> None:
    result = FullBalanceTransferRule().evaluate(
        _frame({"source_row_id": 7}, {"source_row_id": 7})
    )
    assert [item.source_row_id for item in result.evidence] == [7, 7]


@pytest.mark.parametrize("index", [["z", "a"], [100, 7], ["same", "same"]])
def test_evidence_order_uses_input_position(index: list) -> None:
    result = FullBalanceTransferRule().evaluate(_frame({}, {}, index=index))
    assert [item.transaction_id for item in result.evidence] == ["tx-1", "tx-2"]


@pytest.mark.parametrize("value", [None, [], {}, "frame"])
def test_non_dataframe_input_is_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        FullBalanceTransferRule().evaluate(value)


@pytest.mark.parametrize("missing", REQUIRED_COLUMNS)
def test_each_missing_required_column_is_named(missing: str) -> None:
    with pytest.raises(ValueError, match=missing):
        FullBalanceTransferRule().evaluate(_frame().drop(columns=[missing]))


def test_typed_empty_dataframe_is_clean() -> None:
    result = FullBalanceTransferRule().evaluate(pd.DataFrame(columns=REQUIRED_COLUMNS))
    assert (result.triggered, result.score, result.evidence) == (False, 0, ())


def test_completely_empty_dataframe_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="transaction_id"):
        FullBalanceTransferRule().evaluate(pd.DataFrame())


@pytest.mark.parametrize("value", [None, pd.NA, "", "   "])
def test_invalid_candidate_transaction_id(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="transaction_id"):
        FullBalanceTransferRule().evaluate(_frame({"transaction_id": value}))


@pytest.mark.parametrize("field", ["actor_account", "target_account"])
@pytest.mark.parametrize("value", [None, pd.NA, "", "   "])
def test_invalid_candidate_account(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        FullBalanceTransferRule().evaluate(_frame({field: value}))


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_boolean_candidate_source_row_id_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="source_row_id"):
        FullBalanceTransferRule().evaluate(_frame({"source_row_id": value}))


def test_missing_source_row_id_maps_to_none() -> None:
    result = FullBalanceTransferRule().evaluate(_frame({"source_row_id": pd.NA}))
    assert result.evidence[0].source_row_id is None


def test_invalid_candidate_datetime_is_rejected() -> None:
    with pytest.raises(TypeError, match="transaction_datetime"):
        FullBalanceTransferRule().evaluate(_frame({"transaction_datetime": "2026-08-03"}))


def test_nat_datetime_maps_to_none() -> None:
    result = FullBalanceTransferRule().evaluate(_frame({"transaction_datetime": pd.NaT}))
    assert result.evidence[0].transaction_datetime is None


def test_timezone_aware_datetime_is_preserved() -> None:
    value = pd.Timestamp("2026-08-03 10:00:00", tz="Asia/Seoul")
    result = FullBalanceTransferRule().evaluate(_frame({"transaction_datetime": value}))
    assert result.evidence[0].transaction_datetime == value.to_pydatetime()
    assert result.evidence[0].transaction_datetime.tzinfo is not None


def test_python_datetime_is_preserved() -> None:
    value = datetime(2026, 8, 3, 10, tzinfo=timezone.utc)
    result = FullBalanceTransferRule().evaluate(_frame({"transaction_datetime": value}))
    assert result.evidence[0].transaction_datetime == value


def test_malformed_non_candidate_is_ignored() -> None:
    malformed = {
        "transaction_id": pd.NA,
        "source_row_id": True,
        "transaction_datetime": "not-a-date",
        "action_type": "PAYMENT",
        "amount": "not-a-number",
        "balance_before": Decimal("NaN"),
        "actor_account": None,
        "target_account": "",
    }
    result = FullBalanceTransferRule().evaluate(_frame(malformed))
    assert result.triggered is False


def test_detected_path_does_not_mutate_input() -> None:
    frame = _frame({"amount": 1000.0}, {"amount": 1200.0}, index=["b", "a"])
    before = frame.copy(deep=True)
    FullBalanceTransferRule().evaluate(frame)
    assert_frame_equal(frame, before)


class _SuccessRule(BaseRule):
    rule_id = "success"
    rule_name = "Success Rule"
    description = "Returns a valid clean result."

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=False,
            score=0,
            reason="No match.",
        )


class _FailureRule(BaseRule):
    rule_id = "failure"
    rule_name = "Failure Rule"
    description = "Raises a deliberate error."

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        raise RuntimeError("deliberate failure")


def test_rule_engine_returns_plugin_result() -> None:
    registry = RuleRegistry()
    registry.register(FullBalanceTransferRule())
    report = RuleEngine(registry).evaluate(_frame())
    assert [result.rule_id for result in report.results] == ["full_balance_transfer"]
    assert report.errors == ()


def test_plugin_runs_after_previous_rule_failure() -> None:
    registry = RuleRegistry()
    registry.register(_FailureRule())
    registry.register(FullBalanceTransferRule())
    report = RuleEngine(registry).evaluate(_frame())
    assert [result.rule_id for result in report.results] == ["full_balance_transfer"]
    assert [error.rule_id for error in report.errors] == ["failure"]


def test_plugin_failure_is_recorded_and_next_rule_runs() -> None:
    registry = RuleRegistry()
    registry.register(FullBalanceTransferRule())
    registry.register(_SuccessRule())
    report = RuleEngine(registry).evaluate(_frame({"amount": "invalid"}))
    assert [result.rule_id for result in report.results] == ["success"]
    assert report.errors[0].rule_id == "full_balance_transfer"
    assert report.errors[0].error_type == "TypeError"


def test_success_result_order_is_preserved_around_plugin() -> None:
    class _SecondSuccessRule(_SuccessRule):
        rule_id = "second-success"
        rule_name = "Second Success Rule"

    registry = RuleRegistry()
    registry.register(_SuccessRule())
    registry.register(FullBalanceTransferRule())
    registry.register(_SecondSuccessRule())
    report = RuleEngine(registry).evaluate(_frame())
    assert [result.rule_id for result in report.results] == [
        "success",
        "full_balance_transfer",
        "second-success",
    ]
