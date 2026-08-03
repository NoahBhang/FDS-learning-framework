"""Canonical Plugin TransferCashOutRule 계약 테스트."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.dummy_rules import AlwaysFalseRule
from kaggle_bank_fds.src.rules.rule_engine import RuleEngine
from kaggle_bank_fds.src.rules.rule_registry import RuleRegistry
from kaggle_bank_fds.src.rules.rule_result import RuleResult
from kaggle_bank_fds.src.rules.transfer_cash_out_rule import TransferCashOutRule


REQUIRED_COLUMNS = list(TransferCashOutRule.REQUIRED_COLUMNS)


def _row(
    transaction_id: str,
    source_row_id: object,
    step: object,
    action_type: object,
    amount: object,
    actor_account: object,
    target_account: object,
    transaction_datetime: object = pd.NaT,
) -> dict:
    return {
        "transaction_id": transaction_id,
        "source_row_id": source_row_id,
        "step": step,
        "transaction_datetime": transaction_datetime,
        "action_type": action_type,
        "amount": amount,
        "actor_account": actor_account,
        "target_account": target_account,
    }


def _frame(rows=None, index=None) -> pd.DataFrame:
    if rows is None:
        rows = [
            _row(
                "T-1", 10, 5, "TRANSFER", 1_000.0, "SOURCE", "MIDDLE",
                pd.Timestamp("2026-01-01 05:00:00"),
            ),
            _row(
                "C-1", 20, 6, "CASH_OUT", 1_000.0, "MIDDLE", "SINK",
                pd.Timestamp("2026-01-01 06:00:00"),
            ),
        ]
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS, index=index)
    return frame


def _evaluate(rows=None, **rule_kwargs) -> RuleResult:
    return TransferCashOutRule(**rule_kwargs).evaluate(_frame(rows))


def test_metadata_matches_contract() -> None:
    assert TransferCashOutRule.rule_id == "transfer_cash_out"
    assert TransferCashOutRule.rule_name == "Transfer Cash-Out Rule"
    assert TransferCashOutRule.description == (
        "Detects transfers whose receiving account performs a subsequent "
        "cash-out within the configured step window and amount tolerance."
    )


def test_rule_registers_and_can_be_retrieved() -> None:
    registry = RuleRegistry()
    rule = TransferCashOutRule()
    registry.register(rule)
    assert registry.get("transfer_cash_out") is rule


def test_constructor_defaults_and_python_storage_types() -> None:
    rule = TransferCashOutRule()
    assert rule.max_step_gap == 24
    assert type(rule.max_step_gap) is int
    assert rule.amount_tolerance == 0.05
    assert type(rule.amount_tolerance) is float


def test_configuration_properties_are_readonly() -> None:
    rule = TransferCashOutRule()
    with pytest.raises(AttributeError):
        rule.max_step_gap = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        rule.amount_tolerance = 0.1  # type: ignore[misc]


@pytest.mark.parametrize("value", [0, 1, 24, np.int32(3), np.int64(5)])
def test_valid_max_step_gap(value: object) -> None:
    assert TransferCashOutRule(max_step_gap=value).max_step_gap == int(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_boolean_max_step_gap_is_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        TransferCashOutRule(max_step_gap=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1.0, "24", None, 1 + 0j, np.float64(2)])
def test_non_integer_max_step_gap_is_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        TransferCashOutRule(max_step_gap=value)  # type: ignore[arg-type]


def test_negative_max_step_gap_is_rejected() -> None:
    with pytest.raises(ValueError):
        TransferCashOutRule(max_step_gap=-1)


@pytest.mark.parametrize(
    "value", [0, 1, 0.05, np.int64(0), np.float32(0.25), np.float64(0.75)]
)
def test_valid_amount_tolerance(value: object) -> None:
    assert TransferCashOutRule(amount_tolerance=value).amount_tolerance == pytest.approx(  # type: ignore[arg-type]
        float(value)
    )


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_boolean_amount_tolerance_is_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        TransferCashOutRule(amount_tolerance=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["0.05", None, 1 + 0j, np.complex64(0.1)])
def test_non_real_amount_tolerance_is_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        TransferCashOutRule(amount_tolerance=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.01, 1.01])
def test_nonfinite_or_out_of_range_amount_tolerance_is_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        TransferCashOutRule(amount_tolerance=value)


def test_single_pair_result_score_reason_and_evidence_order() -> None:
    result = _evaluate()
    assert result.triggered is True
    assert result.score == 25
    assert [item.transaction_id for item in result.evidence] == ["T-1", "C-1"]
    assert "Detected 1 transfer-to-cash-out pair(s)" in result.reason
    assert "2 unique transaction(s) involved" in result.reason


def test_complete_transfer_evidence_mapping() -> None:
    item = _evaluate().evidence[0]
    assert item.transaction_id == "T-1"
    assert item.source_row_id == 10
    assert item.amount == 1_000.0
    assert item.actor_account == "SOURCE"
    assert item.target_account == "MIDDLE"
    assert item.transaction_datetime == datetime(2026, 1, 1, 5)
    assert item.message == "Transfer matched cash-out transaction C-1."


def test_complete_cashout_evidence_mapping() -> None:
    item = _evaluate().evidence[1]
    assert item.transaction_id == "C-1"
    assert item.source_row_id == 20
    assert item.amount == 1_000.0
    assert item.actor_account == "MIDDLE"
    assert item.target_account == "SINK"
    assert item.transaction_datetime == datetime(2026, 1, 1, 6)
    assert item.message == "Cash-out matched transfer transaction T-1."


@pytest.mark.parametrize("gap", [0, 24])
def test_time_gap_inclusive_boundaries(gap: int) -> None:
    rows = [
        _row("T", 1, 10, "TRANSFER", 100, "A", "M"),
        _row("C", 2, 10 + gap, "CASH_OUT", 100, "M", "Z"),
    ]
    assert _evaluate(rows).triggered is True


@pytest.mark.parametrize("cashout_step", [35, 9])
def test_time_gap_above_maximum_or_reverse_is_not_detected(cashout_step: int) -> None:
    rows = [
        _row("T", 1, 10, "TRANSFER", 100, "A", "M"),
        _row("C", 2, cashout_step, "CASH_OUT", 100, "M", "Z"),
    ]
    result = _evaluate(rows)
    assert result.triggered is False
    assert result.score == 0
    assert result.evidence == ()


def test_custom_max_step_gap_is_applied() -> None:
    rows = [
        _row("T", 1, 1, "TRANSFER", 100, "A", "M"),
        _row("C", 2, 3, "CASH_OUT", 100, "M", "Z"),
    ]
    assert _evaluate(rows, max_step_gap=1).triggered is False
    assert _evaluate(rows, max_step_gap=2).triggered is True


@pytest.mark.parametrize("cashout_amount", [1_000, 950, 1_050, 975, 1_025])
def test_amount_tolerance_equal_boundary_and_inside(cashout_amount: float) -> None:
    rows = [
        _row("T", 1, 1, "TRANSFER", 1_000, "A", "M"),
        _row("C", 2, 2, "CASH_OUT", cashout_amount, "M", "Z"),
    ]
    assert _evaluate(rows).triggered is True


@pytest.mark.parametrize("cashout_amount", [949.9, 1_050.1])
def test_amount_outside_tolerance_is_not_detected(cashout_amount: float) -> None:
    rows = [
        _row("T", 1, 1, "TRANSFER", 1_000, "A", "M"),
        _row("C", 2, 2, "CASH_OUT", cashout_amount, "M", "Z"),
    ]
    assert _evaluate(rows).triggered is False


def test_custom_amount_tolerance_is_applied() -> None:
    rows = [
        _row("T", 1, 1, "TRANSFER", 1_000, "A", "M"),
        _row("C", 2, 2, "CASH_OUT", 900, "M", "Z"),
    ]
    assert _evaluate(rows, amount_tolerance=0.05).triggered is False
    assert _evaluate(rows, amount_tolerance=0.10).triggered is True


@pytest.mark.parametrize(
    ("transfer_amount", "cashout_amount"),
    [(0, 0), (-100, -100), (100, 0), (0, 100), (-100, 100)],
)
def test_nonpositive_amount_is_valid_but_not_detected(
    transfer_amount: float,
    cashout_amount: float,
) -> None:
    rows = [
        _row("T", 1, 1, "TRANSFER", transfer_amount, "A", "M"),
        _row("C", 2, 2, "CASH_OUT", cashout_amount, "M", "Z"),
    ]
    assert _evaluate(rows).triggered is False


@pytest.mark.parametrize("value", [np.nan, pd.NA, np.inf, -np.inf])
def test_missing_or_nonfinite_candidate_amount_is_rejected(value: object) -> None:
    rows = [
        _row("T", 1, 1, "TRANSFER", value, "A", "M"),
        _row("C", 2, 2, "CASH_OUT", 100, "M", "Z"),
    ]
    with pytest.raises(ValueError):
        _evaluate(rows)


@pytest.mark.parametrize("value", [True, np.bool_(True), 1 + 0j, "100"])
def test_invalid_candidate_amount_type_is_rejected(value: object) -> None:
    rows = [
        _row("T", 1, 1, "TRANSFER", value, "A", "M"),
        _row("C", 2, 2, "CASH_OUT", 100, "M", "Z"),
    ]
    with pytest.raises(TypeError):
        _evaluate(rows)


def test_categorical_candidate_amount_is_rejected() -> None:
    frame = _frame()
    frame["amount"] = pd.Series(pd.Categorical([1_000, 1_000]), index=frame.index)
    with pytest.raises(TypeError):
        TransferCashOutRule().evaluate(frame)


def test_one_transfer_to_multiple_cashouts_repeats_pair_evidence() -> None:
    rows = [
        _row("T1", 1, 1, "TRANSFER", 100, "A", "M"),
        _row("C1", 2, 2, "CASH_OUT", 100, "M", "Z1"),
        _row("C2", 3, 3, "CASH_OUT", 100, "M", "Z2"),
    ]
    result = _evaluate(rows)
    assert [item.transaction_id for item in result.evidence] == [
        "T1", "C1", "T1", "C2"
    ]
    assert result.score == 30


def test_multiple_transfers_to_one_cashout_repeats_pair_evidence() -> None:
    rows = [
        _row("T1", 1, 1, "TRANSFER", 100, "A1", "M"),
        _row("T2", 2, 2, "TRANSFER", 100, "A2", "M"),
        _row("C1", 3, 3, "CASH_OUT", 100, "M", "Z"),
    ]
    result = _evaluate(rows)
    assert [item.transaction_id for item in result.evidence] == [
        "T1", "C1", "T2", "C1"
    ]
    assert result.score == 30


def test_independent_pairs_follow_transfer_then_cashout_positions() -> None:
    rows = [
        _row("C2", 4, 4, "CASH_OUT", 200, "M2", "Z2"),
        _row("T1", 1, 1, "TRANSFER", 100, "A1", "M1"),
        _row("T2", 3, 2, "TRANSFER", 200, "A2", "M2"),
        _row("C1", 2, 3, "CASH_OUT", 100, "M1", "Z1"),
    ]
    result = _evaluate(rows)
    assert [item.transaction_id for item in result.evidence] == [
        "T1", "C1", "T2", "C2"
    ]


def test_same_account_time_amount_distinct_transactions_create_all_pairs() -> None:
    rows = [
        _row("T1", 1, 1, "TRANSFER", 100, "A1", "M"),
        _row("T2", 2, 1, "TRANSFER", 100, "A2", "M"),
        _row("C1", 3, 1, "CASH_OUT", 100, "M", "Z1"),
        _row("C2", 4, 1, "CASH_OUT", 100, "M", "Z2"),
    ]
    result = _evaluate(rows)
    assert [item.transaction_id for item in result.evidence] == [
        "T1", "C1", "T1", "C2", "T2", "C1", "T2", "C2"
    ]
    assert result.score == 40


def test_duplicate_candidate_transaction_id_is_rejected() -> None:
    rows = [
        _row("DUP", 1, 1, "TRANSFER", 100, "A", "M"),
        _row("DUP", 2, 2, "CASH_OUT", 100, "M", "Z"),
    ]
    with pytest.raises(ValueError, match="unique"):
        _evaluate(rows)


def test_duplicate_source_row_id_is_allowed_and_preserved() -> None:
    rows = [
        _row("T", 7, 1, "TRANSFER", 100, "A", "M"),
        _row("C", 7, 2, "CASH_OUT", 100, "M", "Z"),
    ]
    result = _evaluate(rows)
    assert [item.source_row_id for item in result.evidence] == [7, 7]


@pytest.mark.parametrize("index", [["a", "b"], [100, 1_000], ["same", "same"]])
def test_dataframe_index_does_not_control_pair_identity_or_order(index: list) -> None:
    result = TransferCashOutRule().evaluate(_frame(index=index))
    assert [item.transaction_id for item in result.evidence] == ["T-1", "C-1"]


def test_non_dataframe_input_is_rejected() -> None:
    with pytest.raises(TypeError):
        TransferCashOutRule().evaluate([])  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", REQUIRED_COLUMNS)
def test_each_missing_required_column_is_named(missing: str) -> None:
    with pytest.raises(ValueError, match=missing):
        TransferCashOutRule().evaluate(_frame().drop(columns=missing))


def test_typed_empty_dataframe_returns_clean_result() -> None:
    result = TransferCashOutRule().evaluate(_frame().iloc[:0])
    assert result.triggered is False
    assert result.score == 0
    assert result.evidence == ()
    assert result.reason == (
        "Detected 0 transfer-to-cash-out pair(s) within 24 step(s) and "
        "5.00% amount tolerance; 0 unique transaction(s) involved."
    )


def test_completely_empty_dataframe_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="transaction_id"):
        TransferCashOutRule().evaluate(pd.DataFrame())


@pytest.mark.parametrize("field,value", [("transaction_id", pd.NA), ("actor_account", None), ("target_account", pd.NA)])
def test_missing_candidate_string_field_is_rejected(field: str, value: object) -> None:
    frame = _frame()
    frame.loc[frame.index[0], field] = value
    with pytest.raises(ValueError):
        TransferCashOutRule().evaluate(frame)


@pytest.mark.parametrize("field", ["transaction_id", "actor_account", "target_account"])
def test_blank_candidate_string_field_is_rejected(field: str) -> None:
    frame = _frame()
    frame.loc[frame.index[0], field] = "   "
    with pytest.raises(ValueError):
        TransferCashOutRule().evaluate(frame)


@pytest.mark.parametrize("field", ["step", "amount"])
def test_missing_candidate_numeric_field_is_rejected(field: str) -> None:
    frame = _frame()
    frame[field] = frame[field].astype("Float64" if field == "amount" else "Int64")
    frame.loc[frame.index[0], field] = pd.NA
    with pytest.raises(ValueError):
        TransferCashOutRule().evaluate(frame)


@pytest.mark.parametrize("value", [True, np.bool_(True), 1.0, 1 + 0j, "1", pd.Timestamp("2026-01-01"), pd.Timedelta(days=1)])
def test_invalid_candidate_step_type_is_rejected(value: object) -> None:
    frame = _frame()
    frame["step"] = frame["step"].astype(object)
    frame.loc[frame.index[0], "step"] = value
    with pytest.raises(TypeError):
        TransferCashOutRule().evaluate(frame)


def test_categorical_candidate_step_is_rejected() -> None:
    frame = _frame()
    frame["step"] = pd.Categorical([1, 2])
    with pytest.raises(TypeError):
        TransferCashOutRule().evaluate(frame)


def test_nullable_integer_and_float_candidates_are_supported() -> None:
    frame = _frame()
    frame["step"] = frame["step"].astype("Int64")
    frame["amount"] = frame["amount"].astype("Float64")
    assert TransferCashOutRule().evaluate(frame).triggered is True


def test_malformed_non_candidate_row_is_ignored() -> None:
    rows = _frame().to_dict("records")
    rows.append(
        _row(
            "", True, "bad", "PAYMENT", "not-an-amount", None, None, "bad-date"
        )
    )
    result = _evaluate(rows)
    assert result.triggered is True


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_boolean_candidate_source_row_id_is_rejected(value: object) -> None:
    frame = _frame()
    frame["source_row_id"] = frame["source_row_id"].astype(object)
    frame.loc[frame.index[0], "source_row_id"] = value
    with pytest.raises(TypeError):
        TransferCashOutRule().evaluate(frame)


def test_missing_candidate_source_row_id_maps_to_none() -> None:
    frame = _frame()
    frame["source_row_id"] = frame["source_row_id"].astype("Int64")
    frame.loc[frame.index[0], "source_row_id"] = pd.NA
    result = TransferCashOutRule().evaluate(frame)
    assert result.evidence[0].source_row_id is None


def test_nat_datetime_maps_to_none() -> None:
    frame = _frame()
    frame["transaction_datetime"] = pd.NaT
    result = TransferCashOutRule().evaluate(frame)
    assert [item.transaction_datetime for item in result.evidence] == [None, None]


def test_invalid_candidate_datetime_is_rejected_even_if_unmatched() -> None:
    rows = [
        _row("T", 1, 1, "TRANSFER", 100, "A", "NO-MATCH", "bad"),
        _row("C", 2, 2, "CASH_OUT", 100, "M", "Z"),
    ]
    with pytest.raises(TypeError):
        _evaluate(rows)


def test_detected_path_does_not_mutate_input_dataframe() -> None:
    frame = _frame(index=["transfer", "cashout"])
    before = frame.copy(deep=True)
    TransferCashOutRule().evaluate(frame)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    ("cashout_count", "score"), [(1, 25), (2, 30), (3, 35), (4, 40), (5, 40)]
)
def test_score_by_unique_pair_count_and_cap(cashout_count: int, score: int) -> None:
    rows = [_row("T", 1, 1, "TRANSFER", 100, "A", "M")]
    rows.extend(
        _row(f"C-{position}", position + 2, position + 2, "CASH_OUT", 100, "M", f"Z-{position}")
        for position in range(cashout_count)
    )
    result = _evaluate(rows)
    assert result.score == score
    assert f"Detected {cashout_count} transfer-to-cash-out pair(s)" in result.reason


class RuntimeFailureRule(BaseRule):
    rule_id = "runtime_failure"
    rule_name = "Runtime Failure Rule"
    description = "Raises a runtime error for engine integration tests."

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        raise RuntimeError("boom")


def test_rule_engine_returns_plugin_result() -> None:
    registry = RuleRegistry()
    registry.register(TransferCashOutRule())
    report = RuleEngine(registry).evaluate(_frame())
    assert report.succeeded_count == 1
    assert report.failed_count == 0
    assert report.results[0].rule_id == "transfer_cash_out"


def test_plugin_runs_after_previous_rule_failure() -> None:
    registry = RuleRegistry()
    registry.register(RuntimeFailureRule())
    registry.register(TransferCashOutRule())
    report = RuleEngine(registry).evaluate(_frame())
    assert [result.rule_id for result in report.results] == ["transfer_cash_out"]
    assert report.errors[0].rule_id == "runtime_failure"


def test_plugin_failure_is_recorded_and_next_rule_runs() -> None:
    registry = RuleRegistry()
    registry.register(TransferCashOutRule())
    registry.register(AlwaysFalseRule())
    malformed = _frame().drop(columns="amount")
    report = RuleEngine(registry).evaluate(malformed)
    assert [result.rule_id for result in report.results] == ["always_false"]
    assert report.errors[0].rule_id == "transfer_cash_out"
    assert report.errors[0].error_type == "ValueError"


def test_success_result_order_is_preserved_around_plugin() -> None:
    registry = RuleRegistry()
    registry.register(AlwaysFalseRule())
    registry.register(TransferCashOutRule())
    report = RuleEngine(registry).evaluate(_frame())
    assert [result.rule_id for result in report.results] == [
        "always_false", "transfer_cash_out"
    ]
