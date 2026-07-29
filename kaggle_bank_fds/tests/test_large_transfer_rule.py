"""LargeTransferRule 계약 및 RuleEngine 통합 테스트."""

from dataclasses import FrozenInstanceError
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.dummy_rules import AlwaysFalseRule
from kaggle_bank_fds.src.rules.evidence_item import EvidenceItem
from kaggle_bank_fds.src.rules.large_transfer_rule import LargeTransferRule
from kaggle_bank_fds.src.rules.rule_engine import RuleEngine
from kaggle_bank_fds.src.rules.rule_registry import RuleRegistry
from kaggle_bank_fds.src.rules.rule_result import RuleResult


def _transactions(
    amounts: pd.Series | list[object],
    transaction_ids: list[object] | None = None,
) -> pd.DataFrame:
    size = len(amounts)
    return pd.DataFrame(
        {
            "transaction_id": transaction_ids
            or [f"TX-{index}" for index in range(size)],
            "amount": amounts,
        }
    )


def test_metadata_matches_contract() -> None:
    assert LargeTransferRule.rule_id == "large_transfer"
    assert LargeTransferRule.rule_name == "Large Transfer Rule"
    assert LargeTransferRule.description == (
        "Detects transactions whose amount is greater than or equal to "
        "the configured threshold."
    )


def test_valid_configuration_is_exposed_read_only() -> None:
    rule = LargeTransferRule(threshold=1_000, score=70)

    assert rule.threshold == 1_000.0
    assert rule.score == 70
    with pytest.raises(AttributeError):
        rule.threshold = 2_000  # type: ignore[misc]


@pytest.mark.parametrize("score", [1, 100])
def test_score_boundaries_are_supported(score: int) -> None:
    rule = LargeTransferRule(threshold=1_000, score=score)

    assert rule.score == score


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), np.inf, -np.inf])
def test_invalid_numeric_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        LargeTransferRule(threshold=threshold, score=50)


@pytest.mark.parametrize("threshold", [True, False, "1000", None])
def test_non_numeric_threshold_is_rejected(threshold: object) -> None:
    with pytest.raises(TypeError):
        LargeTransferRule(threshold=threshold, score=50)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [np.int64(1000), np.float64(1000.5)])
def test_numpy_numeric_threshold_is_supported(threshold: object) -> None:
    rule = LargeTransferRule(threshold=threshold, score=50)  # type: ignore[arg-type]

    assert rule.threshold == float(threshold)


@pytest.mark.parametrize("score", [0, 101, -1])
def test_out_of_range_score_is_rejected(score: int) -> None:
    with pytest.raises(ValueError):
        LargeTransferRule(threshold=1000, score=score)


@pytest.mark.parametrize("score", [True, False, 10.0, "10", None])
def test_non_integer_score_is_rejected(score: object) -> None:
    with pytest.raises(TypeError):
        LargeTransferRule(threshold=1000, score=score)  # type: ignore[arg-type]


def test_numpy_integer_score_is_rejected() -> None:
    with pytest.raises(TypeError):
        LargeTransferRule(threshold=1000, score=np.int64(30))  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, [], {}, "dataframe"])
def test_non_dataframe_input_is_rejected(value: object) -> None:
    rule = LargeTransferRule(1000, 50)

    with pytest.raises(TypeError, match="DataFrame"):
        rule.evaluate(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("frame", "missing_names"),
    [
        (pd.DataFrame({"amount": [1]}), ("transaction_id",)),
        (pd.DataFrame({"transaction_id": ["TX-1"]}), ("amount",)),
        (pd.DataFrame(), ("transaction_id", "amount")),
    ],
)
def test_missing_required_columns_are_named(
    frame: pd.DataFrame,
    missing_names: tuple[str, ...],
) -> None:
    rule = LargeTransferRule(1000, 50)

    with pytest.raises(ValueError) as error:
        rule.evaluate(frame)

    assert all(name in str(error.value) for name in missing_names)


@pytest.mark.parametrize(
    "amounts",
    [
        pd.Series(["1000"], dtype="string"),
        pd.Series(["1000"], dtype=object),
        pd.Series(["1000"], dtype="category"),
        pd.Series([True], dtype=bool),
        pd.Series([True], dtype="boolean"),
    ],
)
def test_non_numeric_amount_dtype_is_rejected(amounts: pd.Series) -> None:
    rule = LargeTransferRule(1000, 50)

    with pytest.raises(TypeError, match="amount"):
        rule.evaluate(_transactions(amounts))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_amount_dtype_is_rejected_by_rule_contract(
    dtype: type[np.complexfloating],
) -> None:
    amounts = pd.Series([1000 + 0j], dtype=dtype)

    with pytest.raises(TypeError) as error:
        LargeTransferRule(1000, 50).evaluate(_transactions(amounts))

    message = str(error.value)
    assert "amount" in message
    assert "complex" in message
    assert "float() argument" not in message


@pytest.mark.parametrize(
    "amounts",
    [
        pd.Series([pd.Timestamp("2026-01-01")]),
        pd.Series([pd.Timedelta(days=1)]),
    ],
)
def test_datetime_and_timedelta_amount_dtypes_are_rejected(
    amounts: pd.Series,
) -> None:
    with pytest.raises(TypeError, match="amount"):
        LargeTransferRule(1000, 50).evaluate(_transactions(amounts))


def test_mixed_bool_object_amount_dtype_is_rejected() -> None:
    amounts = pd.Series([True, 1000], dtype=object)

    with pytest.raises(TypeError, match="amount"):
        LargeTransferRule(1000, 50).evaluate(_transactions(amounts))


@pytest.mark.parametrize(
    ("amount", "triggered"),
    [
        (999.99, False),
        (1000.0, True),
        (1000.01, True),
        (0.0, False),
        (-1000.0, False),
        (np.nan, False),
    ],
)
def test_detection_boundary_and_ignored_values(
    amount: float,
    triggered: bool,
) -> None:
    result = LargeTransferRule(1000, 60).evaluate(
        _transactions([amount])
    )

    assert result.triggered is triggered
    assert result.score == (60 if triggered else 0)


def test_nullable_missing_amount_is_ignored() -> None:
    frame = _transactions(pd.Series([pd.NA], dtype="Float64"))

    result = LargeTransferRule(1000, 60).evaluate(frame)

    assert result.triggered is False
    assert result.evidence == ()


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
def test_all_null_nullable_amount_is_clean(dtype: str) -> None:
    frame = _transactions(pd.Series([pd.NA, pd.NA], dtype=dtype))

    result = LargeTransferRule(1000, 60).evaluate(frame)

    assert result.triggered is False
    assert result.score == 0
    assert result.evidence == ()


@pytest.mark.parametrize("amount", [np.inf, -np.inf])
def test_infinite_amount_fails_entire_evaluation(amount: float) -> None:
    frame = _transactions([2000.0, amount])

    with pytest.raises(ValueError, match=r"amount.*유한"):
        LargeTransferRule(1000, 60).evaluate(frame)


def test_multiple_detections_keep_input_order_and_fixed_score() -> None:
    frame = _transactions(
        [3000.0, 100.0, 2000.0, 1000.0],
        ["TX-3", "TX-0", "TX-2", "TX-1"],
    )
    frame.index = [40, 10, 30, 20]

    result = LargeTransferRule(1000, 67).evaluate(frame)

    assert result.triggered is True
    assert result.score == 67
    assert [item.transaction_id for item in result.evidence] == [
        "TX-3",
        "TX-2",
        "TX-1",
    ]
    assert len(result.evidence) == 3


def test_complete_evidence_mapping() -> None:
    timestamp = pd.Timestamp("2026-01-02 03:04:05")
    frame = pd.DataFrame(
        {
            "transaction_id": [" TX-1 "],
            "amount": [1500],
            "source_row_id": [np.int64(42)],
            "actor_account": ["A-1"],
            "target_account": ["B-1"],
            "transaction_datetime": [timestamp],
        }
    )

    result = LargeTransferRule(1000, 75).evaluate(frame)
    item = result.evidence[0]

    assert isinstance(item, EvidenceItem)
    assert item.transaction_id == "TX-1"
    assert item.amount == 1500.0
    assert item.source_row_id == 42
    assert item.actor_account == "A-1"
    assert item.target_account == "B-1"
    assert item.transaction_datetime == timestamp.to_pydatetime()


def test_missing_optional_columns_are_supported() -> None:
    result = LargeTransferRule(1000, 75).evaluate(
        _transactions([1500])
    )
    item = result.evidence[0]

    assert item.source_row_id is None
    assert item.actor_account is None
    assert item.target_account is None
    assert item.transaction_datetime is None


def test_nullable_optional_values_become_none() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": ["TX-1"],
            "amount": pd.Series([1500], dtype="Int64"),
            "source_row_id": [pd.NA],
            "actor_account": [pd.NA],
            "target_account": [np.nan],
            "transaction_datetime": [pd.NaT],
        }
    )

    item = LargeTransferRule(1000, 75).evaluate(frame).evidence[0]

    assert item.source_row_id is None
    assert item.actor_account is None
    assert item.target_account is None
    assert item.transaction_datetime is None


@pytest.mark.parametrize("source_row_id", [True, False, np.bool_(True)])
def test_bool_source_row_id_on_detected_row_is_rejected(
    source_row_id: object,
) -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": ["TX-1"],
            "amount": pd.Series([1500], dtype="Int64"),
            "source_row_id": [source_row_id],
        }
    )

    with pytest.raises(TypeError) as error:
        LargeTransferRule(1000, 75).evaluate(frame)

    message = str(error.value)
    assert "source_row_id" in message
    assert "bool" in message


def test_bool_source_row_id_on_undetected_row_is_ignored() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": ["TX-1"],
            "amount": pd.Series([10], dtype="Int64"),
            "source_row_id": [True],
        }
    )

    result = LargeTransferRule(1000, 75).evaluate(frame)

    assert result.triggered is False
    assert result.evidence == ()


@pytest.mark.parametrize("transaction_id", [None, pd.NA, "", "   "])
def test_invalid_detected_transaction_id_is_rejected(
    transaction_id: object,
) -> None:
    frame = _transactions([1500], [transaction_id])

    with pytest.raises((TypeError, ValueError), match="transaction_id"):
        LargeTransferRule(1000, 75).evaluate(frame)


@pytest.mark.parametrize("transaction_id", [None, pd.NA, "", "   "])
def test_invalid_undetected_transaction_id_is_ignored(
    transaction_id: object,
) -> None:
    frame = _transactions([10], [transaction_id])

    result = LargeTransferRule(1000, 75).evaluate(frame)

    assert result.triggered is False


def test_result_contract_for_no_detection() -> None:
    result = LargeTransferRule(1000, 75).evaluate(_transactions([10]))

    assert isinstance(result, RuleResult)
    assert result.rule_id == "large_transfer"
    assert result.rule_name == "Large Transfer Rule"
    assert result.triggered is False
    assert result.score == 0
    assert result.evidence == ()
    assert "0" in result.reason
    assert "1000.0" in result.reason


def test_result_contract_for_detection_and_tuple_evidence() -> None:
    result = LargeTransferRule(1000, 75).evaluate(
        _transactions([1000, 2000])
    )

    assert result.triggered is True
    assert result.score == 75
    assert isinstance(result.evidence, tuple)
    assert len(result.evidence) == 2
    assert "2" in result.reason
    assert "1000.0" in result.reason


def test_result_and_evidence_are_immutable() -> None:
    result = LargeTransferRule(1000, 75).evaluate(_transactions([1500]))

    with pytest.raises(FrozenInstanceError):
        result.score = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.evidence[0].amount = 1.0  # type: ignore[misc]


def test_empty_dataframe_with_required_columns_is_clean() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": pd.Series(dtype="string"),
            "amount": pd.Series(dtype="Float64"),
        }
    )

    result = LargeTransferRule(1000, 75).evaluate(frame)

    assert result.triggered is False
    assert result.score == 0
    assert result.evidence == ()


@pytest.mark.parametrize(
    "amounts",
    [
        pd.Series([999, 1000], dtype="Int64"),
        pd.Series([999.0, 1000.0], dtype="Float64"),
        pd.Series([999, 1000], dtype=np.int32),
        pd.Series([999, 1000], dtype=np.int64),
        pd.Series([999.0, 1000.0], dtype=np.float32),
        pd.Series([999.0, 1000.0], dtype=np.float64),
    ],
)
def test_supported_numeric_dtypes(amounts: pd.Series) -> None:
    result = LargeTransferRule(1000, 75).evaluate(_transactions(amounts))

    assert [item.transaction_id for item in result.evidence] == ["TX-1"]


def test_evaluate_does_not_mutate_input_dataframe() -> None:
    frame = pd.DataFrame(
        {
            "transaction_id": ["TX-1", "TX-2"],
            "amount": [1000, pd.NA],
            "actor_account": ["A", pd.NA],
        },
        index=[7, 3],
    ).astype(
        {
            "transaction_id": "string",
            "amount": "Float64",
            "actor_account": "string",
        }
    )
    before = frame.copy(deep=True)

    result = LargeTransferRule(1000, 75).evaluate(frame)

    assert result.triggered is True
    assert len(result.evidence) == 1
    assert_frame_equal(frame, before, check_dtype=True, check_like=False)


def test_rule_can_be_registered_and_run_by_engine() -> None:
    registry = RuleRegistry()
    rule = LargeTransferRule(1000, 75)
    registry.register(rule)

    report = RuleEngine(registry).evaluate(_transactions([1500]))

    assert registry.get("large_transfer") is rule
    assert report.succeeded_count == 1
    assert report.failed_count == 0
    assert report.results[0].rule_id == "large_transfer"
    assert report.results[0].triggered is True


class _FailingRule(BaseRule):
    rule_id = "failing"
    rule_name = "Failing Rule"
    description = "RuleEngine 오류 격리 테스트용 Rule"

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        raise RuntimeError("expected failure")


class _FinalSuccessRule(AlwaysFalseRule):
    rule_id = "final_success"
    rule_name = "Final Success Rule"
    description = "RuleEngine 순서 복구 테스트용 Rule"


def test_large_transfer_runs_after_another_rule_fails() -> None:
    registry = RuleRegistry()
    registry.register(_FailingRule())
    registry.register(LargeTransferRule(1000, 75))

    report = RuleEngine(registry).evaluate(_transactions([1500]))

    assert report.succeeded_count == 1
    assert report.failed_count == 1
    assert report.results[0].rule_id == "large_transfer"
    assert report.errors[0].rule_id == "failing"


def test_large_transfer_input_error_is_recorded_and_next_rule_runs() -> None:
    registry = RuleRegistry()
    registry.register(AlwaysFalseRule())
    registry.register(LargeTransferRule(1000, 75))
    registry.register(_FinalSuccessRule())

    report = RuleEngine(registry).evaluate(
        pd.DataFrame({"transaction_id": ["TX-1"]})
    )

    assert report.succeeded_count == 2
    assert report.failed_count == 1
    assert [result.rule_id for result in report.results] == [
        "always_false",
        "final_success",
    ]
    assert report.errors[0].rule_id == "large_transfer"
    assert report.errors[0].error_type == "ValueError"
