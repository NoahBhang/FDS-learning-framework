from pathlib import Path

import pandas as pd


APP_DIR = Path(__file__).resolve().parent

OUTPUT_PATH = (
    APP_DIR
    / "data"
    / "generic_bank_test.csv"
)


def main() -> None:
    test_df = pd.DataFrame(
        {
            "거래일자": [
                "2026-07-01",
                "2026-07-01",
                "2026-07-02",
                "2026-07-03",
            ],
            "거래시간": [
                "09:10:00",
                "14:35:00",
                "02:20:00",
                "18:05:00",
            ],
            "출금액": [
                "1,000,000",
                "0",
                "5,000,000",
                "0",
            ],
            "입금액": [
                "0",
                "2,500,000",
                "0",
                "300,000",
            ],
            "잔액": [
                "9,000,000",
                "11,500,000",
                "6,500,000",
                "6,800,000",
            ],
            "적요": [
                "계좌이체",
                "급여",
                "ATM출금",
                "입금",
            ],
            "상대방": [
                "홍길동",
                "주식회사 예시",
                "",
                "김철수",
            ],
            "상대계좌": [
                "110-***-123456",
                "",
                "",
                "020-***-987654",
            ],
            "은행명": [
                "테스트은행",
                "테스트은행",
                "테스트은행",
                "테스트은행",
            ],
        }
    )

    test_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"범용 은행 테스트 CSV 생성 완료: "
        f"{OUTPUT_PATH}"
    )
    print(f"행 수: {len(test_df):,}건")


if __name__ == "__main__":
    main()