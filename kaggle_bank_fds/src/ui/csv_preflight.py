"""Pure PaySim CSV parsing and user-correctable preflight validation."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import csv
from pathlib import Path
from typing import BinaryIO

import pandas as pd


PAYSIM_REQUIRED_COLUMNS = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
)


class CsvPreflightError(ValueError):
    """Base error for a CSV issue the user can correct."""


class UnsupportedCsvFileError(CsvPreflightError):
    pass


class CsvFileTooLargeError(CsvPreflightError):
    pass


class CsvRowLimitExceededError(CsvPreflightError):
    pass


class MissingRequiredColumnsError(CsvPreflightError):
    def __init__(self, missing_columns: tuple[str, ...]) -> None:
        self.missing_columns = missing_columns
        super().__init__("필수 CSV 컬럼이 없습니다: " + ", ".join(missing_columns))


@dataclass(frozen=True, slots=True)
class ParsedCsvUpload:
    dataframe: pd.DataFrame
    filename: str
    file_size_bytes: int
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    preview: pd.DataFrame


def parse_and_validate_paysim_csv(
    source: bytes | bytearray | BinaryIO,
    *,
    filename: str,
    file_size_bytes: int,
    max_rows: int = 10_000,
    max_bytes: int = 20 * 1024 * 1024,
    preview_rows: int = 20,
) -> ParsedCsvUpload:
    normalized_filename = _validate_filename(filename)
    _positive_integer(file_size_bytes, "file_size_bytes", allow_zero=True)
    _positive_integer(max_rows, "max_rows")
    _positive_integer(max_bytes, "max_bytes")
    _positive_integer(preview_rows, "preview_rows")
    if file_size_bytes == 0:
        raise CsvPreflightError("CSV 파일이 비어 있습니다.")
    if file_size_bytes > max_bytes:
        raise CsvFileTooLargeError("CSV 파일이 허용된 크기를 초과했습니다.")

    data = _read_bytes_preserving_position(source)
    if not data:
        raise CsvPreflightError("CSV 파일이 비어 있습니다.")
    if len(data) > max_bytes:
        raise CsvFileTooLargeError("CSV 파일이 허용된 크기를 초과했습니다.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CsvPreflightError("CSV 파일은 UTF-8 인코딩이어야 합니다.") from exc
    _validate_header(text)
    try:
        dataframe = pd.read_csv(BytesIO(data), encoding="utf-8", low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise CsvPreflightError("CSV 내용을 해석할 수 없습니다.") from exc
    if dataframe.empty:
        raise CsvPreflightError("CSV에 분석할 거래 행이 없습니다.")
    if len(dataframe) > max_rows:
        raise CsvRowLimitExceededError("CSV 행 수가 허용된 한도를 초과했습니다.")
    missing = tuple(column for column in PAYSIM_REQUIRED_COLUMNS if column not in dataframe)
    if missing:
        raise MissingRequiredColumnsError(missing)
    parsed = dataframe.copy(deep=True)
    return ParsedCsvUpload(
        dataframe=parsed,
        filename=normalized_filename,
        file_size_bytes=file_size_bytes,
        row_count=len(parsed),
        column_count=len(parsed.columns),
        columns=tuple(parsed.columns),
        preview=parsed.head(preview_rows).copy(deep=True),
    )


def _validate_filename(filename: object) -> str:
    if not isinstance(filename, str):
        raise TypeError("filename must be a string.")
    normalized = filename.strip()
    if not normalized:
        raise UnsupportedCsvFileError("CSV 파일명이 비어 있습니다.")
    if Path(normalized).suffix.lower() != ".csv":
        raise UnsupportedCsvFileError("CSV 확장자 파일만 업로드할 수 있습니다.")
    return normalized


def _positive_integer(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a bool-free integer.")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _read_bytes_preserving_position(source: bytes | bytearray | BinaryIO) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if not hasattr(source, "read"):
        raise TypeError("source must be bytes or a binary file-like object.")
    position = None
    if hasattr(source, "tell") and hasattr(source, "seek"):
        try:
            position = source.tell()
            source.seek(0)
        except (OSError, ValueError):
            position = None
    try:
        data = source.read()
    finally:
        if position is not None:
            source.seek(position)
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("source must provide bytes.")
    return bytes(data)


def _validate_header(text: str) -> None:
    try:
        header = next(csv.reader(text.splitlines()))
    except (StopIteration, csv.Error) as exc:
        raise CsvPreflightError("CSV header를 해석할 수 없습니다.") from exc
    if len(header) != len(set(header)):
        raise CsvPreflightError("CSV에 중복된 컬럼 이름이 있습니다.")
