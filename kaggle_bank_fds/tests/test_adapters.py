"""Canonical Transaction Adapter의 독립 회귀 테스트."""

import importlib

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.adapters import (
    AdapterRegistry,
    CANONICAL_COLUMNS,
    GenericCSVAdapter,
    PaySimAdapter,
)
from kaggle_bank_fds.src.adapters.canonical_transaction_schema import (
    CANONICAL_DTYPES,
)


def make_paysim_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 100_000,
                "nameOrig": "C100",
                "oldbalanceOrg": 100_000,
                "newbalanceOrig": 0,
                "nameDest": "C200",
                "oldbalanceDest": 0,
                "newbalanceDest": 100_000,
            },
            {
                "step": 2,
                "type": "CASH_OUT",
                "amount": 99_000,
                "nameOrig": "C200",
                "oldbalanceOrg": 100_000,
                "newbalanceOrig": 1_000,
                "nameDest": "C300",
                "oldbalanceDest": 0,
                "newbalanceDest": 99_000,
            },
        ]
    )


def make_generic_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "거래일자": "2026-07-01",
                "거래시간": "09:10:00",
                "출금액": "1,000,000",
                "입금액": pd.NA,
                "거래 후 잔액": "9,000,000",
                "상대방": "홍길동",
                "상대계좌번호": "110-001",
                "적요": "계좌이체",
                "은행명": "테스트은행",
            },
            {
                "거래일자": "2026-07-02",
                "거래시간": "14:35:00",
                "출금액": pd.NA,
                "입금액": "2,500,000",
                "거래 후 잔액": "11,500,000",
                "상대방": "주식회사 예시",
                "상대계좌번호": "220-002",
                "적요": "급여",
                "은행명": "테스트은행",
            },
        ]
    )


def test_package_imports_succeed() -> None:
    modules = [
        "kaggle_bank_fds.src.adapters.paysim_adapter",
        "kaggle_bank_fds.src.adapters.generic_csv_adapter",
        "kaggle_bank_fds.src.adapters.adapter_registry",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name) is not None


def test_paysim_adapter_can_handle_paysim() -> None:
    assert PaySimAdapter().can_handle(make_paysim_df())


def test_generic_adapter_can_handle_generic_csv() -> None:
    assert GenericCSVAdapter().can_handle(make_generic_df())


def test_generic_adapter_does_not_capture_paysim() -> None:
    assert not GenericCSVAdapter().can_handle(make_paysim_df())


@pytest.mark.parametrize(
    ("dataframe_factory", "expected_adapter"),
    [
        (make_paysim_df, PaySimAdapter),
        (make_generic_df, GenericCSVAdapter),
    ],
)
def test_registry_selects_expected_adapter(
    dataframe_factory,
    expected_adapter,
) -> None:
    selected = AdapterRegistry().detect(dataframe_factory())
    assert isinstance(selected, expected_adapter)


def test_registry_rejects_unsupported_format() -> None:
    unsupported = pd.DataFrame(
        {"이름": ["홍길동"], "주소": ["서울"]}
    )

    with pytest.raises(
        ValueError,
        match="처리 가능한 Adapter",
    ):
        AdapterRegistry().detect(unsupported)


def test_both_adapters_return_identical_column_order() -> None:
    paysim_result = PaySimAdapter().transform(make_paysim_df())
    generic_result = GenericCSVAdapter().transform(make_generic_df())

    assert paysim_result.columns.tolist() == CANONICAL_COLUMNS
    assert generic_result.columns.tolist() == CANONICAL_COLUMNS


def test_both_adapters_follow_dtype_contract() -> None:
    results = [
        PaySimAdapter().transform(make_paysim_df()),
        GenericCSVAdapter().transform(make_generic_df()),
    ]

    expected = {
        column: dtype
        for column, dtype in CANONICAL_DTYPES.items()
    }
    for result in results:
        actual = {
            column: str(dtype)
            for column, dtype in result.dtypes.items()
        }
        assert actual == expected


@pytest.mark.parametrize(
    ("adapter", "dataframe_factory"),
    [
        (PaySimAdapter(), make_paysim_df),
        (GenericCSVAdapter(), make_generic_df),
    ],
)
def test_transaction_ids_are_deterministic(
    adapter,
    dataframe_factory,
) -> None:
    source = dataframe_factory()

    first = adapter.transform(source)
    second = adapter.transform(source)

    assert first["transaction_id"].equals(
        second["transaction_id"]
    )
    assert first["transaction_id"].is_unique


@pytest.mark.parametrize(
    ("adapter", "dataframe_factory"),
    [
        (PaySimAdapter(), make_paysim_df),
        (GenericCSVAdapter(), make_generic_df),
    ],
)
def test_string_input_index_is_supported(
    adapter,
    dataframe_factory,
) -> None:
    source = dataframe_factory()
    source.index = [f"row-{index}" for index in range(len(source))]

    result = adapter.transform(source)

    assert result["source_row_id"].tolist() == list(
        range(len(source))
    )


def test_generic_adapter_rejects_invalid_amount() -> None:
    source = make_generic_df()
    source.loc[source.index[1], "입금액"] = "금액오류"

    with pytest.raises(
        ValueError,
        match=r"source_row_id=\[1\].*금액오류",
    ):
        GenericCSVAdapter().transform(source)


def test_generic_adapter_rejects_empty_deposit_and_withdrawal() -> None:
    source = make_generic_df()
    source.loc[source.index[0], ["출금액", "입금액"]] = pd.NA

    with pytest.raises(
        ValueError,
        match=r"정확히 한 값만 양수.*source_row_id=\[0\]",
    ):
        GenericCSVAdapter().transform(source)


def test_generic_adapter_rejects_positive_deposit_and_withdrawal() -> None:
    source = make_generic_df()
    source.loc[source.index[0], "입금액"] = "500,000"

    with pytest.raises(
        ValueError,
        match=r"정확히 한 값만 양수.*source_row_id=\[0\]",
    ):
        GenericCSVAdapter().transform(source)


@pytest.mark.parametrize("amount_column", ["입금액", "출금액"])
def test_generic_adapter_rejects_non_numeric_split_amount(
    amount_column,
) -> None:
    source = make_generic_df()
    source.loc[source.index[0], amount_column] = "금액오류"

    with pytest.raises(
        ValueError,
        match=r"source_row_id=\[0\].*금액오류",
    ):
        GenericCSVAdapter().transform(source)


def test_paysim_adapter_rejects_missing_required_column() -> None:
    source = make_paysim_df().drop(columns=["nameDest"])

    with pytest.raises(
        ValueError,
        match="nameDest",
    ):
        PaySimAdapter().transform(source)


def test_paysim_adapter_rejects_missing_account_value() -> None:
    source = make_paysim_df()
    source.loc[source.index[0], "nameOrig"] = pd.NA

    with pytest.raises(
        ValueError,
        match=r"source_row_id=\[0\]",
    ):
        PaySimAdapter().transform(source)


def test_account_columns_never_contain_nan_string() -> None:
    results = [
        PaySimAdapter().transform(make_paysim_df()),
        GenericCSVAdapter().transform(make_generic_df()),
    ]

    for result in results:
        for column in ["actor_account", "target_account"]:
            values = result[column].dropna().str.lower()
            assert not values.eq("nan").any()


def test_generic_missing_strings_remain_nullable() -> None:
    source = make_generic_df()
    source.loc[source.index[0], "상대방"] = pd.NA
    source.loc[source.index[0], "상대계좌번호"] = pd.NA
    source.loc[source.index[0], "적요"] = pd.NA

    result = GenericCSVAdapter().transform(source)

    assert pd.isna(result.loc[0, "actor_account"])
    assert pd.isna(result.loc[0, "target_account"])
    assert pd.isna(result.loc[0, "counterparty_name"])
    assert pd.isna(result.loc[0, "description"])


def test_registry_transform_does_not_write_stdout(capsys) -> None:
    AdapterRegistry().transform(make_paysim_df())
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.parametrize(
    ("adapter", "dataframe_factory"),
    [
        (PaySimAdapter(), make_paysim_df),
        (GenericCSVAdapter(), make_generic_df),
    ],
)
def test_transform_does_not_mutate_source_dataframe(
    adapter,
    dataframe_factory,
) -> None:
    source = dataframe_factory()
    before = source.copy(deep=True)

    adapter.transform(source)

    assert_frame_equal(source, before)
