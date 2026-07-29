from pathlib import Path
import sys

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from adapters.adapter_registry import AdapterRegistry


def test_paysim(registry: AdapterRegistry) -> None:
    """
    PaySim CSV를 자동 감지하고 변환한다.
    """

    csv_path = (
        APP_DIR
        / "data"
        / "paysim_test_10000.csv"
    )

    original_df = pd.read_csv(csv_path)

    selected_adapter = registry.detect(original_df)

    print("=" * 70)
    print("1. PaySim 자동 감지 테스트")
    print("=" * 70)
    print(f"원본 파일: {csv_path.name}")
    print(f"원본 행 수: {len(original_df):,}건")
    print(f"선택된 Adapter: {selected_adapter.adapter_name}")

    standard_df = registry.transform(original_df)

    print(f"변환 행 수: {len(standard_df):,}건")
    print(
        standard_df[
            [
                "action_type",
                "amount",
                "actor_account",
                "target_account",
                "source_format",
            ]
        ].head()
    )

    assert selected_adapter.adapter_name == "paysim"
    assert len(standard_df) == len(original_df)
    assert standard_df["source_format"].eq("PAYSIM").all()

    print("PaySim Registry 테스트 완료")
    print()


def test_generic_csv(registry: AdapterRegistry) -> None:
    """
    일반 은행 CSV를 자동 감지하고 변환한다.
    """

    csv_path = (
        APP_DIR
        / "data"
        / "generic_bank_test.csv"
    )

    original_df = pd.read_csv(csv_path)

    selected_adapter = registry.detect(original_df)

    print("=" * 70)
    print("2. Generic CSV 자동 감지 테스트")
    print("=" * 70)
    print(f"원본 파일: {csv_path.name}")
    print(f"원본 행 수: {len(original_df):,}건")
    print(f"선택된 Adapter: {selected_adapter.adapter_name}")

    standard_df = registry.transform(original_df)

    print(f"변환 행 수: {len(standard_df):,}건")
    print(
        standard_df[
            [
                "transaction_datetime",
                "action_type",
                "amount",
                "description",
                "source_format",
            ]
        ].to_string(index=False)
    )

    assert (
        selected_adapter.adapter_name
        == "generic_bank_csv"
    )
    assert len(standard_df) == len(original_df)
    assert (
        standard_df["source_format"]
        .eq("GENERIC_CSV")
        .all()
    )

    print("Generic CSV Registry 테스트 완료")
    print()


def test_unsupported_csv(
    registry: AdapterRegistry,
) -> None:
    """
    지원하지 않는 열 구조에서 오류가 발생하는지 검사한다.
    """

    unsupported_df = pd.DataFrame(
        {
            "이름": ["홍길동", "김철수"],
            "주소": ["서울", "대전"],
        }
    )

    print("=" * 70)
    print("3. 미지원 CSV 예외 처리 테스트")
    print("=" * 70)

    try:
        registry.detect(unsupported_df)

    except ValueError as error:
        print(f"예상된 오류 발생: {error}")

    else:
        raise AssertionError(
            "미지원 CSV인데 Adapter가 선택되었습니다."
        )

    print("미지원 CSV 예외 처리 테스트 완료")
    print()


def main() -> None:
    registry = AdapterRegistry()

    test_paysim(registry)
    test_generic_csv(registry)
    test_unsupported_csv(registry)

    print("=" * 70)
    print("Adapter Registry 전체 검증 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
