from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from adapters.paysim_adapter import PaySimAdapter


def main() -> None:
    csv_path = (
        Path(__file__).resolve().parent
        / "data"
        / "paysim_test_10000.csv"
    )

    original_df = pd.read_csv(csv_path)

    adapter = PaySimAdapter()

    print("=" * 60)
    print("PaySim Adapter 테스트")
    print("=" * 60)

    print(f"원본 행 수: {len(original_df):,}건")
    print(f"처리 가능 여부: {adapter.can_handle(original_df)}")

    standardized_df = adapter.transform(original_df)

    print(f"변환 행 수: {len(standardized_df):,}건")
    print()
    print("표준 열 목록:")
    print(standardized_df.columns.tolist())
    print()
    print("변환 결과 일부:")
    print(standardized_df.head())

    assert len(standardized_df) == len(original_df)
    assert adapter.can_handle(original_df)
    assert standardized_df["amount"].notna().all()

    print()
    print("PaySim Adapter 검증 완료")


if __name__ == "__main__":
    main()