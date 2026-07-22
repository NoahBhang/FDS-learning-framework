import pandas as pd

from bankruptcy_fds.src.rules.split_transfer import SplitTransferRule


def make_tx(rows):
    return pd.DataFrame(rows)


def evaluate(rows, filing_date="2026-06-30"):
    rule = SplitTransferRule(
        window_days=30,
        min_count=5,
        total_amount_threshold=5_000_000,
    )
    df = make_tx(rows)
    return rule.evaluate(
        df,
        filing_date=pd.Timestamp(filing_date),
        debtor_id="D001",
    )


def base_row(tx_id, date, amount, receiver="김OO", tx_type="송금"):
    return {
        "transaction_id": tx_id,
        "debtor_id": "D001",
        "transaction_date": pd.Timestamp(date),
        "amount": amount,
        "sender": "D001",
        "receiver": receiver,
        "relation_type": "가족",
        "transaction_type": tx_type,
    }


def test_detects_exactly_five_transfers():
    rows = [
        base_row(f"T{i}", f"2026-06-{10+i:02d}", 1_000_000)
        for i in range(5)
    ]

    result = evaluate(rows)

    assert result.is_suspicious is True


def test_ignores_four_transfers():
    rows = [
        base_row(f"T{i}", f"2026-06-{10+i:02d}", 1_250_000)
        for i in range(4)
    ]

    result = evaluate(rows)

    assert result.is_suspicious is False


def test_detects_exactly_threshold_amount():
    rows = [
        base_row(f"T{i}", f"2026-06-{10+i:02d}", 1_000_000)
        for i in range(5)
    ]

    result = evaluate(rows)

    assert result.is_suspicious is True


def test_ignores_amount_below_threshold():
    rows = [
        base_row(f"T{i}", f"2026-06-{10+i:02d}", 999_999)
        for i in range(5)
    ]

    result = evaluate(rows)

    assert result.is_suspicious is False


def test_includes_30_days_before_filing_date():
    rows = [
        base_row(f"T{i}", f"2026-06-{1+i:02d}", 1_000_000)
        for i in range(5)
    ]

    result = evaluate(rows, filing_date="2026-07-01")

    assert result.is_suspicious is True


def test_excludes_31_days_before_filing_date():
    rows = [
        base_row("T1", "2026-05-31", 1_000_000),
        base_row("T2", "2026-06-02", 1_000_000),
        base_row("T3", "2026-06-03", 1_000_000),
        base_row("T4", "2026-06-04", 1_000_000),
        base_row("T5", "2026-06-05", 1_000_000),
    ]

    result = evaluate(rows, filing_date="2026-07-01")

    assert result.is_suspicious is False
