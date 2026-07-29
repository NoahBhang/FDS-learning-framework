"""
은행별 거래내역 파일을 공통 거래 스키마로 변환하기 위한
어댑터 추상 클래스이다.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseBankAdapter(ABC):
    """
    모든 은행 거래내역 어댑터가 따라야 하는 공통 계약이다.

    은행마다 열 이름과 파일 구조는 다르지만,
    분석 엔진에는 동일한 표준 DataFrame을 전달한다.
    """

    adapter_name: str = "base"
    bank_name: str = "unknown"

    @abstractmethod
    def can_handle(self, df: pd.DataFrame) -> bool:
        """
        현재 DataFrame을 이 어댑터가 처리할 수 있는지 판단한다.

        Returns
        -------
        bool
            처리할 수 있으면 True, 아니면 False
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        은행 원본 DataFrame을 표준 거래 스키마로 변환한다.

        Returns
        -------
        pd.DataFrame
            표준화된 거래 DataFrame
        """
        raise NotImplementedError

    def validate_required_columns(
        self,
        df: pd.DataFrame,
        required_columns: set[str],
    ) -> None:
        """
        변환에 필요한 필수 열이 존재하는지 검사한다.
        """

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))

            raise ValueError(
                f"{self.adapter_name} 어댑터가 요구하는 열이 없습니다: "
                f"{missing_text}"
            )