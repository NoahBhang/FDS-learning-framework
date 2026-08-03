"""Legacy bank Rule 회귀 테스트."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.models.predictor import DEFAULT_RULES, predict
from kaggle_bank_fds.src.rules.bank_fraud_rules import (
    FullBalanceTransferRule,
    TransferCashOutRule,
)
from shared.rules.base_fraud_rule import RuleResult


def _row(
    *,
    step: int,
    transaction_type: str,
    amount: float,
    origin: str,
    destination: str,
    old_balance: float = 2_000,
) -> dict:
    return {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "nameOrig": origin,
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": old_balance - amount,
        "nameDest": destination,
        "oldbalanceDest": 0,
        "newbalanceDest": amount,
    }


def _pair(
    *,
    step_gap: int = 1,
    transfer_amount: float = 1_000,
    cashout_amount: float = 1_000,
    indexes=(101, 909),
    transfer_destination: str = "MIDDLE",
    cashout_origin: str = "MIDDLE",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(
                step=10,
                transaction_type="TRANSFER",
                amount=transfer_amount,
                origin="SOURCE",
                destination=transfer_destination,
            ),
            _row(
                step=10 + step_gap,
                transaction_type="CASH_OUT",
                amount=cashout_amount,
                origin=cashout_origin,
                destination="SINK",
            ),
        ],
        index=list(indexes),
    )


def test_evidence_uses_original_indexes_instead_of_merge_range_index() -> None:
    result = TransferCashOutRule().evaluate(_pair(indexes=(101, 909)))

    assert result.evidence_ids == [101, 909]
    assert result.evidence_ids != [0]


def test_transfer_and_cashout_indexes_are_both_evidence() -> None:
    result = TransferCashOutRule().evaluate(_pair(indexes=(7, 42)))

    assert result.evidence_ids == [7, 42]


def test_non_contiguous_integer_indexes_are_preserved() -> None:
    result = TransferCashOutRule().evaluate(_pair(indexes=(100, 10_000)))

    assert result.evidence_ids == [100, 10_000]


def test_string_indexes_are_preserved_without_integer_coercion() -> None:
    result = TransferCashOutRule().evaluate(
        _pair(indexes=("tx-row-a", "cash-row-z"))
    )

    assert result.evidence_ids == ["tx-row-a", "cash-row-z"]
    assert all(isinstance(value, str) for value in result.evidence_ids)


def test_pair_evidence_order_is_transfer_then_cashout() -> None:
    result = TransferCashOutRule().evaluate(_pair(indexes=("transfer", "cashout")))

    assert result.evidence_ids == ["transfer", "cashout"]


def test_duplicate_index_values_are_removed_by_first_seen_equality() -> None:
    result = TransferCashOutRule().evaluate(_pair(indexes=("same", "same")))

    assert result.evidence_ids == ["same"]


def test_boolean_index_values_follow_legacy_list_contract_without_conversion() -> None:
    result = TransferCashOutRule().evaluate(_pair(indexes=(True, False)))

    assert result.evidence_ids == [True, False]
    assert [type(value) for value in result.evidence_ids] == [bool, bool]


def test_unhashable_index_values_are_preserved_without_set_dependency() -> None:
    frame = _pair()
    frame.index = pd.Index([["transfer"], ["cashout"]])

    result = TransferCashOutRule().evaluate(frame)

    assert result.evidence_ids == [["transfer"], ["cashout"]]


def test_repeated_transfer_index_is_kept_once_in_first_seen_order() -> None:
    frame = pd.DataFrame(
        [
            _row(
                step=1,
                transaction_type="TRANSFER",
                amount=1_000,
                origin="SOURCE",
                destination="MIDDLE",
            ),
            _row(
                step=2,
                transaction_type="CASH_OUT",
                amount=1_000,
                origin="MIDDLE",
                destination="SINK-1",
            ),
            _row(
                step=3,
                transaction_type="CASH_OUT",
                amount=1_000,
                origin="MIDDLE",
                destination="SINK-2",
            ),
        ],
        index=["transfer", "cash-1", "cash-2"],
    )

    result = TransferCashOutRule().evaluate(frame)

    assert result.evidence_ids == ["transfer", "cash-1", "cash-2"]


def test_repeated_cashout_index_is_kept_once_in_first_seen_order() -> None:
    frame = pd.DataFrame(
        [
            _row(
                step=1,
                transaction_type="TRANSFER",
                amount=1_000,
                origin="SOURCE-1",
                destination="MIDDLE",
            ),
            _row(
                step=2,
                transaction_type="TRANSFER",
                amount=1_000,
                origin="SOURCE-2",
                destination="MIDDLE",
            ),
            _row(
                step=3,
                transaction_type="CASH_OUT",
                amount=1_000,
                origin="MIDDLE",
                destination="SINK",
            ),
        ],
        index=["transfer-1", "transfer-2", "cashout"],
    )

    result = TransferCashOutRule().evaluate(frame)

    assert result.evidence_ids == ["transfer-1", "cashout", "transfer-2"]


@pytest.mark.parametrize("step_gap", [0, 24])
def test_step_gap_boundaries_are_detected(step_gap: int) -> None:
    result = TransferCashOutRule().evaluate(_pair(step_gap=step_gap))

    assert result.is_suspicious is True


def test_step_gap_above_boundary_is_not_detected() -> None:
    result = TransferCashOutRule().evaluate(_pair(step_gap=25))

    assert result.is_suspicious is False


@pytest.mark.parametrize("cashout_amount", [950, 1_050, 975, 1_025])
def test_amount_tolerance_boundary_and_inside_are_detected(
    cashout_amount: float,
) -> None:
    result = TransferCashOutRule().evaluate(
        _pair(cashout_amount=cashout_amount)
    )

    assert result.is_suspicious is True


@pytest.mark.parametrize("cashout_amount", [949.9, 1_050.1])
def test_amount_outside_tolerance_is_not_detected(cashout_amount: float) -> None:
    result = TransferCashOutRule().evaluate(
        _pair(cashout_amount=cashout_amount)
    )

    assert result.is_suspicious is False


def test_unrelated_accounts_are_not_detected() -> None:
    result = TransferCashOutRule().evaluate(
        _pair(transfer_destination="A", cashout_origin="B")
    )

    assert result.is_suspicious is False


def test_non_detection_returns_clean_legacy_rule_result() -> None:
    result = TransferCashOutRule().evaluate(_pair(step_gap=25))

    assert isinstance(result, RuleResult)
    assert result.rule_name == "transfer_cash_out"
    assert result.risk_type == "이체 후 즉시 현금화 의심"
    assert result.is_suspicious is False
    assert result.risk_score == 0
    assert result.evidence_ids == []


def test_detection_returns_suspicious_legacy_rule_result_and_list_evidence() -> None:
    result = TransferCashOutRule().evaluate(_pair())

    assert isinstance(result, RuleResult)
    assert result.is_suspicious is True
    assert isinstance(result.evidence_ids, list)


def test_one_pair_score_and_reason_are_unchanged() -> None:
    result = TransferCashOutRule().evaluate(_pair())

    assert result.risk_score == 25
    assert "총 1건" in result.reason


@pytest.mark.parametrize(
    ("cashout_count", "expected_score"),
    [(2, 30), (3, 35), (4, 40), (5, 40)],
)
def test_multiple_pair_score_and_cap_are_unchanged(
    cashout_count: int,
    expected_score: int,
) -> None:
    rows = [
        _row(
            step=1,
            transaction_type="TRANSFER",
            amount=1_000,
            origin="SOURCE",
            destination="MIDDLE",
        )
    ]
    indexes = ["transfer"]
    for position in range(cashout_count):
        rows.append(
            _row(
                step=2 + position,
                transaction_type="CASH_OUT",
                amount=1_000,
                origin="MIDDLE",
                destination=f"SINK-{position}",
            )
        )
        indexes.append(f"cashout-{position}")

    result = TransferCashOutRule().evaluate(pd.DataFrame(rows, index=indexes))

    assert result.risk_score == expected_score
    assert f"총 {cashout_count}건" in result.reason


def test_evaluate_does_not_mutate_input_dataframe() -> None:
    frame = _pair(indexes=("transfer", "cashout"))
    before = frame.copy(deep=True)

    TransferCashOutRule().evaluate(frame)

    assert_frame_equal(frame, before)


def test_full_balance_transfer_rule_still_detects_original_contract() -> None:
    frame = pd.DataFrame(
        [
            _row(
                step=1,
                transaction_type="TRANSFER",
                amount=1_000,
                origin="SOURCE",
                destination="TARGET",
                old_balance=1_000,
            )
        ],
        index=["full-balance"],
    )

    result = FullBalanceTransferRule().evaluate(frame)

    assert result.is_suspicious is True
    assert result.risk_score == 20
    assert result.evidence_ids == ["full-balance"]


def test_predictor_default_rules_are_unchanged() -> None:
    assert [type(rule) for rule in DEFAULT_RULES] == [
        TransferCashOutRule,
        FullBalanceTransferRule,
    ]


def test_legacy_predict_still_returns_transfer_cashout_result() -> None:
    result = predict(_pair(indexes=("transfer", "cashout")))

    assert "transfer_cash_out" in result["triggered_rules"]
    assert result["details"]["transfer_cash_out"]["evidence_ids"] == [
        "transfer",
        "cashout",
    ]
    assert result["details"]["skipped_rules"] == {}
