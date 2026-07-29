"""
은행 거래 Adapter가 반환하는 Canonical Transaction Schema 계약이다.

이 모듈은 컬럼 순서, dtype, 필수 컬럼 검증과 재현 가능한 분석용
surrogate transaction ID 생성을 한 곳에서 관리한다.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd


CANONICAL_COLUMNS = [
    "transaction_id",
    "source_row_id",
    "step",
    "transaction_datetime",
    "action_type",
    "amount",
    "actor_account",
    "target_account",
    "counterparty_name",
    "balance_before",
    "balance_after",
    "target_balance_before",
    "target_balance_after",
    "description",
    "bank_name",
    "source_format",
]

CANONICAL_DTYPES = {
    "transaction_id": "string",
    "source_row_id": "Int64",
    "step": "Int64",
    "transaction_datetime": "datetime64[ns]",
    "action_type": "string",
    "amount": "Float64",
    "actor_account": "string",
    "target_account": "string",
    "counterparty_name": "string",
    "balance_before": "Float64",
    "balance_after": "Float64",
    "target_balance_before": "Float64",
    "target_balance_after": "Float64",
    "description": "string",
    "bank_name": "string",
    "source_format": "string",
}

TRANSACTION_ID_FIELDS = [
    "source_format",
    "bank_name",
    "source_row_id",
    "transaction_datetime",
    "action_type",
    "amount",
    "actor_account",
    "target_account",
]


def validate_canonical_columns(df: pd.DataFrame) -> None:
    """Canonical Schema의 모든 컬럼이 존재하는지 검사한다."""

    missing_columns = set(CANONICAL_COLUMNS) - set(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Canonical Transaction Schema 필수 컬럼이 없습니다: "
            f"{missing_text}"
        )


def _hash_value(value: object) -> str:
    """결측치와 pandas scalar를 결정적인 해시 입력 문자열로 바꾼다."""

    if pd.isna(value):
        return "<NA>"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def generate_transaction_ids(
    df: pd.DataFrame,
    prefix: str,
) -> pd.Series:
    """
    현재 입력 파일 안에서 재현 가능한 분석용 surrogate ID를 생성한다.

    이 ID는 금융기관 원장의 절대적인 거래 식별자가 아니다. 동일한 입력
    DataFrame을 동일한 행 순서로 변환했을 때 같은 값을 얻기 위한 ID다.
    """

    missing_fields = set(TRANSACTION_ID_FIELDS) - set(df.columns)
    if missing_fields:
        missing_text = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"transaction_id 생성 필드가 없습니다: {missing_text}"
        )

    normalized_prefix = prefix.strip().upper()
    transaction_ids: list[str] = []

    for row in df[TRANSACTION_ID_FIELDS].itertuples(
        index=False,
        name=None,
    ):
        payload = json.dumps(
            [_hash_value(value) for value in row],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()[:24]
        transaction_ids.append(
            f"{normalized_prefix}_{digest}"
        )

    return pd.Series(
        transaction_ids,
        index=df.index,
        dtype="string",
    )


def normalize_canonical_dataframe(
    df: pd.DataFrame,
    transaction_id_prefix: str,
) -> pd.DataFrame:
    """컬럼 순서와 dtype을 Canonical Transaction Schema에 맞춘다."""

    validate_canonical_columns(df)
    normalized = df.loc[:, CANONICAL_COLUMNS].copy()

    for column, dtype in CANONICAL_DTYPES.items():
        if column == "transaction_id":
            continue
        if column == "transaction_datetime":
            normalized[column] = pd.to_datetime(
                normalized[column],
                errors="coerce",
            )
        else:
            normalized[column] = normalized[column].astype(dtype)

    normalized["transaction_id"] = generate_transaction_ids(
        normalized,
        prefix=transaction_id_prefix,
    )
    normalized["transaction_id"] = normalized[
        "transaction_id"
    ].astype(CANONICAL_DTYPES["transaction_id"])

    return normalized.loc[:, CANONICAL_COLUMNS]
