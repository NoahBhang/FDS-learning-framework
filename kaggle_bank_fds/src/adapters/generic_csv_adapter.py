"""
일반적인 국내 은행 거래내역 CSV를
공통 거래 스키마로 변환하는 범용 어댑터이다.

실제 은행별 양식이 확보되기 전까지 사용하는 기본형이며,
열 이름을 확실히 식별할 수 있을 때만 자동 변환한다.
"""

from typing import ClassVar

import pandas as pd

from .base_bank_adapter import BaseBankAdapter
from .canonical_transaction_schema import (
    normalize_canonical_dataframe,
)


class GenericCSVAdapter(BaseBankAdapter):
    """
    국내 은행 거래내역에서 자주 사용되는 열 이름을 탐색하여
    표준 거래 스키마로 변환한다.
    """

    adapter_name = "generic_bank_csv"
    bank_name = "Generic Bank"
    source_format = "GENERIC_CSV"

    COLUMN_ALIASES: ClassVar[dict[str, list[str]]] = {
        "transaction_date": [
            "거래일자",
            "거래일",
            "일자",
            "처리일자",
        ],
        "transaction_time": [
            "거래시간",
            "시간",
            "처리시간",
        ],
        "transaction_datetime": [
            "거래일시",
            "처리일시",
            "거래일자시간",
        ],
        "withdrawal_amount": [
            "출금액",
            "출금금액",
            "찾으신금액",
            "지급금액",
        ],
        "deposit_amount": [
            "입금액",
            "입금금액",
            "맡기신금액",
        ],
        "amount": [
            "거래금액",
            "금액",
        ],
        "balance_after": [
            "거래후잔액",
            "거래 후 잔액",
            "잔액",
            "처리후잔액",
        ],
        "description": [
            "적요",
            "기재내용",
            "거래내용",
            "내용",
        ],
        "counterparty_name": [
            "상대방",
            "거래상대방",
            "받는분",
            "보낸분",
        ],
        "counterparty_account": [
            "상대계좌",
            "상대계좌번호",
            "거래상대계좌",
        ],
        "bank_name": [
            "은행명",
            "금융기관",
            "상대은행",
        ],
    }

    def _normalize_column_name(
        self,
        column_name: object,
    ) -> str:
        """
        열 이름의 앞뒤 공백과 줄바꿈을 제거하여 비교한다.
        """

        return (
            str(column_name)
            .strip()
            .replace("\n", "")
            .replace("\r", "")
        )

    def _build_column_lookup(
        self,
        df: pd.DataFrame,
    ) -> dict[str, str]:
        """
        정규화된 열 이름과 실제 원본 열 이름의 대응표를 만든다.
        """

        return {
            self._normalize_column_name(column): str(column)
            for column in df.columns
        }

    def _find_column(
        self,
        df: pd.DataFrame,
        standard_name: str,
    ) -> str | None:
        """
        표준 항목에 대응하는 실제 원본 열 이름을 찾는다.
        """

        column_lookup = self._build_column_lookup(df)

        for alias in self.COLUMN_ALIASES.get(
            standard_name,
            [],
        ):
            normalized_alias = self._normalize_column_name(
                alias
            )

            if normalized_alias in column_lookup:
                return column_lookup[normalized_alias]

        return None

    def can_handle(
        self,
        df: pd.DataFrame,
    ) -> bool:
        """
        최소한 날짜와 금액 관련 열을 식별할 수 있는지 검사한다.
        """

        datetime_column = self._find_column(
            df,
            "transaction_datetime",
        )

        date_column = self._find_column(
            df,
            "transaction_date",
        )

        amount_column = self._find_column(
            df,
            "amount",
        )

        withdrawal_column = self._find_column(
            df,
            "withdrawal_amount",
        )

        deposit_column = self._find_column(
            df,
            "deposit_amount",
        )

        has_datetime = (
            datetime_column is not None
            or date_column is not None
        )

        has_amount = (
            amount_column is not None
            or withdrawal_column is not None
            or deposit_column is not None
        )

        return has_datetime and has_amount

    def _clean_amount_series(
        self,
        series: pd.Series,
        source_row_id: pd.Series,
        allow_missing: bool,
    ) -> pd.Series:
        """
        쉼표, 원화 기호, 공백이 포함된 금액을 숫자로 변환한다.

        입금액/출금액처럼 서로 보완하는 열에서는 빈 셀을 0으로 취급한다.
        그 밖의 숫자로 해석할 수 없는 값은 원본 행 위치와 함께 거부한다.
        """

        cleaned_series = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₩", "", regex=False)
            .str.replace("원", "", regex=False)
            .str.strip()
        )

        missing_mask = (
            series.isna()
            | cleaned_series.str.lower().isin(
                {"", "nan", "none", "<na>", "-"}
            )
        )
        numeric_series = pd.to_numeric(
            cleaned_series,
            errors="coerce",
        )
        invalid_mask = numeric_series.isna() & ~missing_mask

        if not allow_missing:
            invalid_mask = invalid_mask | missing_mask

        if invalid_mask.any():
            invalid_rows = source_row_id[invalid_mask].tolist()
            invalid_values = series[invalid_mask].tolist()
            raise ValueError(
                "Generic CSV amount를 숫자로 변환할 수 없습니다. "
                f"source_row_id={invalid_rows}, "
                f"values={invalid_values}"
            )

        if allow_missing:
            numeric_series = numeric_series.mask(
                missing_mask,
                0.0,
            )

        return numeric_series.astype("Float64")

    def _create_transaction_datetime(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        거래일시 열 또는 거래일자와 거래시간 열을 조합한다.
        """

        datetime_column = self._find_column(
            df,
            "transaction_datetime",
        )

        if datetime_column is not None:
            return pd.to_datetime(
                df[datetime_column],
                errors="coerce",
            )

        date_column = self._find_column(
            df,
            "transaction_date",
        )

        time_column = self._find_column(
            df,
            "transaction_time",
        )

        if date_column is None:
            return pd.Series(
                pd.NaT,
                index=df.index,
                dtype="datetime64[ns]",
            )

        if time_column is None:
            return pd.to_datetime(
                df[date_column],
                errors="coerce",
            )

        combined_datetime = (
            df[date_column].astype(str).str.strip()
            + " "
            + df[time_column].astype(str).str.strip()
        )

        return pd.to_datetime(
            combined_datetime,
            errors="coerce",
        )

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        일반 은행 CSV를 표준 거래 스키마로 변환한다.
        """

        if not self.can_handle(df):
            raise ValueError(
                "범용 CSV 어댑터가 날짜 열과 금액 열을 "
                "확실하게 식별하지 못했습니다. "
                "사용자 열 매핑 또는 은행별 어댑터가 필요합니다."
            )

        source_row_id = pd.Series(
            range(len(df)),
            index=df.index,
            dtype="Int64",
        )
        amount_column = self._find_column(
            df,
            "amount",
        )

        withdrawal_column = self._find_column(
            df,
            "withdrawal_amount",
        )

        deposit_column = self._find_column(
            df,
            "deposit_amount",
        )

        if amount_column is not None:
            amount_series = self._clean_amount_series(
                df[amount_column],
                source_row_id=source_row_id,
                allow_missing=False,
            )

            action_type = pd.Series(
                "UNKNOWN",
                index=df.index,
                dtype="object",
            )

        else:
            withdrawal_amount = (
                self._clean_amount_series(
                    df[withdrawal_column],
                    source_row_id=source_row_id,
                    allow_missing=True,
                )
                if withdrawal_column is not None
                else pd.Series(
                    0.0,
                    index=df.index,
                )
            )

            deposit_amount = (
                self._clean_amount_series(
                    df[deposit_column],
                    source_row_id=source_row_id,
                    allow_missing=True,
                )
                if deposit_column is not None
                else pd.Series(
                    0.0,
                    index=df.index,
                    dtype="Float64",
                )
            )

            both_empty = (
                (withdrawal_amount <= 0)
                & (deposit_amount <= 0)
            )
            both_positive = (
                (withdrawal_amount > 0)
                & (deposit_amount > 0)
            )
            invalid_direction = both_empty | both_positive

            if invalid_direction.any():
                invalid_rows = source_row_id[
                    invalid_direction
                ].tolist()
                raise ValueError(
                    "Generic CSV 입금액과 출금액 중 정확히 한 값만 "
                    "양수여야 합니다. "
                    f"source_row_id={invalid_rows}"
                )

            amount_series = (
                withdrawal_amount
                + deposit_amount
            )

            action_type = pd.Series(
                "UNKNOWN",
                index=df.index,
                dtype="object",
            )

            action_type.loc[
                withdrawal_amount > 0
            ] = "WITHDRAWAL"

            action_type.loc[
                deposit_amount > 0
            ] = "DEPOSIT"

        balance_column = self._find_column(
            df,
            "balance_after",
        )

        description_column = self._find_column(
            df,
            "description",
        )

        counterparty_name_column = self._find_column(
            df,
            "counterparty_name",
        )

        counterparty_account_column = self._find_column(
            df,
            "counterparty_account",
        )

        bank_name_column = self._find_column(
            df,
            "bank_name",
        )

        standardized_df = pd.DataFrame(
            {
                "transaction_id": pd.NA,
                "source_row_id": source_row_id,
                "step": pd.NA,
                "transaction_datetime":
                    self._create_transaction_datetime(df),
                "action_type": action_type,
                "amount": amount_series,
                "actor_account": pd.NA,
                "target_account": (
                    df[counterparty_account_column]
                    .astype("string")
                    if counterparty_account_column is not None
                    else pd.NA
                ),
                "balance_before": pd.NA,
                "balance_after": (
                    self._clean_amount_series(
                        df[balance_column],
                        source_row_id=source_row_id,
                        allow_missing=True,
                    )
                    if balance_column is not None
                    else pd.NA
                ),
                "target_balance_before": pd.NA,
                "target_balance_after": pd.NA,
                "counterparty_name": (
                    df[counterparty_name_column]
                    .astype("string")
                    if counterparty_name_column is not None
                    else pd.NA
                ),
                "description": (
                    df[description_column]
                    .astype("string")
                    if description_column is not None
                    else pd.NA
                ),
                "bank_name": (
                    df[bank_name_column]
                    .fillna(self.bank_name)
                    .astype("string")
                    if bank_name_column is not None
                    else self.bank_name
                ),
                "source_format": self.source_format,
            },
            index=df.index,
        )

        return normalize_canonical_dataframe(
            standardized_df,
            transaction_id_prefix="GENERIC",
        )
