"""Contracts for pure PaySim CSV preflight."""

from io import BytesIO

import pandas as pd
import pytest

from kaggle_bank_fds.src.ui.csv_preflight import (
    CsvFileTooLargeError,
    CsvPreflightError,
    CsvRowLimitExceededError,
    MissingRequiredColumnsError,
    PAYSIM_REQUIRED_COLUMNS,
    UnsupportedCsvFileError,
    parse_and_validate_paysim_csv,
)


def _csv(*, columns=PAYSIM_REQUIRED_COLUMNS, rows=1):
    header = ",".join(columns)
    values = {
        "step": "1", "type": "TRANSFER", "amount": "100", "nameOrig": "A",
        "oldbalanceOrg": "100", "newbalanceOrig": "0", "nameDest": "B",
        "oldbalanceDest": "0", "newbalanceDest": "100", "extra": "x",
    }
    line = ",".join(values.get(column, "x") for column in columns)
    return (header + "\n" + "\n".join(line for _ in range(rows)) + "\n").encode()


def _parse(data, **kwargs):
    options = dict(filename="input.csv")
    options["file_size_bytes"] = kwargs.pop(
        "file_size_bytes", len(data) if isinstance(data, (bytes, bytearray)) else len(data.getvalue())
    )
    options.update(kwargs)
    return parse_and_validate_paysim_csv(data, **options)


def test_valid_csv_extra_columns_order_and_uppercase_extension():
    columns = ("extra", *reversed(PAYSIM_REQUIRED_COLUMNS))
    result = _parse(_csv(columns=columns), filename="INPUT.CSV", file_size_bytes=len(_csv(columns=columns)))
    assert result.row_count == 1 and result.column_count == 10
    assert result.columns == columns and result.filename == "INPUT.CSV"


@pytest.mark.parametrize("filename", ["input.txt", "input", " "])
def test_non_csv_or_blank_filename_is_rejected(filename):
    data = _csv()
    with pytest.raises(UnsupportedCsvFileError):
        _parse(data, filename=filename, file_size_bytes=len(data))


def test_empty_bytes_empty_dataframe_malformed_and_invalid_utf8():
    with pytest.raises(CsvPreflightError, match="비어"):
        _parse(b"", file_size_bytes=0)
    with pytest.raises(CsvPreflightError, match="거래 행"):
        _parse((",".join(PAYSIM_REQUIRED_COLUMNS) + "\n").encode())
    with pytest.raises(CsvPreflightError):
        _parse(b'a,b\n"unterminated')
    with pytest.raises(CsvPreflightError, match="UTF-8"):
        _parse(b"\xff\xfe")


def test_file_and_row_limits():
    data = _csv(rows=2)
    with pytest.raises(CsvFileTooLargeError):
        _parse(data, max_bytes=len(data) - 1)
    with pytest.raises(CsvRowLimitExceededError):
        _parse(data, max_rows=1)


def test_missing_one_and_multiple_required_columns():
    one = tuple(value for value in PAYSIM_REQUIRED_COLUMNS if value != "amount")
    with pytest.raises(MissingRequiredColumnsError) as error:
        _parse(_csv(columns=one))
    assert error.value.missing_columns == ("amount",)
    multiple = tuple(value for value in one if value != "step")
    with pytest.raises(MissingRequiredColumnsError) as error:
        _parse(_csv(columns=multiple))
    assert error.value.missing_columns == ("step", "amount")


def test_duplicate_column_is_rejected_before_pandas_mangling():
    columns = (*PAYSIM_REQUIRED_COLUMNS, "step")
    with pytest.raises(CsvPreflightError, match="중복"):
        _parse(_csv(columns=columns))


def test_file_pointer_is_restored_and_source_not_mutated():
    data = _csv(); source = BytesIO(data); source.seek(3)
    result = _parse(source, file_size_bytes=len(data))
    assert source.tell() == 3 and source.getvalue() == data
    assert result.row_count == 1


def test_dataframe_and_preview_are_defensively_separate():
    data = _csv(rows=3); result = _parse(data, preview_rows=2)
    result.dataframe.loc[0, "amount"] = 999
    assert result.preview.loc[0, "amount"] == 100
    assert len(result.preview) == 2


@pytest.mark.parametrize("name", ["max_rows", "max_bytes", "preview_rows"])
@pytest.mark.parametrize("value", [True, 0, -1])
def test_limits_reject_bool_and_nonpositive(name, value):
    data = _csv()
    error = TypeError if value is True else ValueError
    with pytest.raises(error):
        _parse(data, **{name: value})
