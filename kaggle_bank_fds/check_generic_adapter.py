from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from adapters.generic_csv_adapter import (
    GenericCSVAdapter,
)


def main() -> None:
    csv_path = (
        Path(__file__).resolve().parent
        / "data"
        / "generic_bank_test.csv"
    )

    original_df = pd.read_csv(csv_path)

    adapter = GenericCSVAdapter()

    print("=" * 70)
    print("Generic CSV Adapter 테스트")
    print("=" * 70)

    print(f"원본 행 수: {len(original_df):,}건")
    print(
        f"처리 가능 여부: "
        f"{adapter.can_handle(original_df)}"
    )

    standardized_df = adapter.transform(
        original_df
    )

    print(
        f"변환 행 수: "
        f"{len(standardized_df):,}건"
    )
    print()

    print("표준 열 목록:")
    print(standardized_df.columns.tolist())
    print()

    print("핵심 변환 결과:")
    print(
        standardized_df[
            [
                "transaction_datetime",
                "action_type",
                "amount",
                "balance_after",
                "counterparty_name",
                "description",
                "bank_name",
            ]
        ].to_string(index=False)
    )

    assert adapter.can_handle(original_df)
    assert len(standardized_df) == 4
    assert (
        standardized_df["amount"].tolist()
        == [
            1_000_000.0,
            2_500_000.0,
            5_000_000.0,
            300_000.0,
        ]
    )
    assert (
        standardized_df["action_type"].tolist()
        == [
            "WITHDRAWAL",
            "DEPOSIT",
            "WITHDRAWAL",
            "DEPOSIT",
        ]
    )

    print()
    print("Generic CSV Adapter 검증 완료")


if __name__ == "__main__":
    main()