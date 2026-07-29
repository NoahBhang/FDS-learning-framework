"""
등록된 Adapter를 관리하고,
DataFrame에 맞는 Adapter를 자동 선택한다.
"""

from __future__ import annotations

import pandas as pd

from .base_bank_adapter import BaseBankAdapter
from .paysim_adapter import PaySimAdapter
from .generic_csv_adapter import GenericCSVAdapter


class AdapterRegistry:
    """
    Adapter들을 등록하고
    가장 적합한 Adapter를 찾아주는 클래스이다.
    """

    def __init__(self) -> None:

        self.adapters: list[BaseBankAdapter] = [
            PaySimAdapter(),
            GenericCSVAdapter(),
        ]

    def detect(
        self,
        df: pd.DataFrame,
    ) -> BaseBankAdapter:
        """
        DataFrame을 처리할 수 있는 Adapter를 찾아 반환한다.
        """

        for adapter in self.adapters:

            if adapter.can_handle(df):
                return adapter

        raise ValueError(
            "처리 가능한 Adapter를 찾지 못했습니다."
        )

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        적절한 Adapter를 자동 선택한 후
        표준 스키마로 변환한다.
        """

        adapter = self.detect(df)

        return adapter.transform(df)
