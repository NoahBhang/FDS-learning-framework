from pathlib import Path

import pandas as pd


APP_DIR = Path(__file__).resolve().parent

SOURCE_PATH = (
    APP_DIR
    / "data"
    / "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_PATH = (
    APP_DIR
    / "data"
    / "paysim_test_10000.csv"
)


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"원본 CSV 파일을 찾을 수 없습니다: {SOURCE_PATH}"
        )

    test_df = pd.read_csv(
        SOURCE_PATH,
        nrows=10_000,
    )

    test_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"테스트 CSV 생성 완료: {OUTPUT_PATH}"
    )
    print(
        f"행 수: {len(test_df):,}건"
    )
    print(
        f"파일 크기: "
        f"{OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f}MB"
    )


if __name__ == "__main__":
    main()
