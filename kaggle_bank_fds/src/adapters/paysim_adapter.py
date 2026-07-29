"""
PaySim 거래 데이터를 표준 은행 거래 스키마로 변환한다.
"""

import pandas as pd

from .base_bank_adapter import BaseBankAdapter
from .canonical_transaction_schema import (
    normalize_canonical_dataframe,
)


class PaySimAdapter(BaseBankAdapter):
    """
    Kaggle PaySim 형식 전용 어댑터이다.
    """

    adapter_name = "paysim"
    bank_name = "PaySim"
    source_format = "PAYSIM"

    REQUIRED_COLUMNS = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
    }

    def can_handle(self, df: pd.DataFrame) -> bool:
        """
        PaySim의 필수 열이 모두 존재하는지 검사한다.
        """

        return self.REQUIRED_COLUMNS.issubset(df.columns)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PaySim 열을 공통 거래 스키마로 변환한다.
        """

        self.validate_required_columns(
            df=df,
            required_columns=self.REQUIRED_COLUMNS,
        )

        source_row_id = pd.Series(
            range(len(df)),
            index=df.index,
            dtype="Int64",
        )
        invalid_accounts = (
            df["nameOrig"].isna()
            | df["nameDest"].isna()
            | df["nameOrig"].astype("string").str.strip().eq("")
            | df["nameDest"].astype("string").str.strip().eq("")
        )
        if invalid_accounts.any():
            invalid_rows = source_row_id[invalid_accounts].tolist()
            raise ValueError(
                "PaySim 필수 계좌 컬럼에 결측값이 있습니다. "
                f"source_row_id={invalid_rows}"
            )

        amount = pd.to_numeric(
            df["amount"],
            errors="coerce",
        )
        if amount.isna().any():
            invalid_rows = source_row_id[amount.isna()].tolist()
            invalid_values = df.loc[
                amount.isna(),
                "amount",
            ].tolist()
            raise ValueError(
                "PaySim amount를 숫자로 변환할 수 없습니다. "
                f"source_row_id={invalid_rows}, "
                f"values={invalid_values}"
            )

        standardized_df = pd.DataFrame(
            {
                "transaction_id": pd.NA,
                "source_row_id": source_row_id,
                "step": pd.to_numeric(
                    df["step"],
                    errors="coerce",
                ),
                "transaction_datetime": pd.NaT,
                "action_type": df["type"].astype("string"),
                "amount": amount,
                "actor_account": df["nameOrig"].astype("string"),
                "target_account": df["nameDest"].astype("string"),
                "counterparty_name": pd.NA,
                "balance_before": pd.to_numeric(
                    df["oldbalanceOrg"],
                    errors="coerce",
                ),
                "balance_after": pd.to_numeric(
                    df["newbalanceOrig"],
                    errors="coerce",
                ),
                "target_balance_before": pd.to_numeric(
                    df["oldbalanceDest"],
                    errors="coerce",
                ),
                "target_balance_after": pd.to_numeric(
                    df["newbalanceDest"],
                    errors="coerce",
                ),
                "description": pd.NA,
                "bank_name": self.bank_name,
                "source_format": self.source_format,
            },
            index=df.index,
        )

        return normalize_canonical_dataframe(
            standardized_df,
            transaction_id_prefix=self.source_format,
        )
