import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bankruptcy_fds.src.rules.velocity_spike import VelocitySpikeRule


def make_tx(rows):
    return pd.DataFrame(rows)


def tx_row(tx_id, date, amount, receiver="김00", tx_type="송금"):
    return {
        "transaction_id": tx_id,
        "debtor_id": "D001",
        "transaction_date": pd.Timestamp(date),
        "amount": amount,
        "sender": "D001",
        "receiver": receiver,
        "relation_type": "지인",
        "transaction_type": tx_type,
    }


def test_velocity_spike_detects_spike_week():
    filing_date = pd.Timestamp("2026-06-30")

    # 기준(baseline) 5개 주 — 각 주에 20만 원짜리 거래 1건씩만 있다.
    baseline = [
        tx_row(f"B{i}", date, 200_000)
        for i, date in enumerate([
            "2026-04-13", "2026-04-20", "2026-04-27", "2026-05-04", "2026-05-11",
        ])
    ]
    # 급증 주 — 2026-06-22(월)~06-26(금), 6건, 각 100만 원.
    spike = [
        tx_row(f"S{i}", date, 1_000_000)
        for i, date in enumerate([
            "2026-06-22", "2026-06-23", "2026-06-24",
            "2026-06-25", "2026-06-25", "2026-06-26",
        ])
    ]

    tx = make_tx(baseline + spike)
    rule = VelocitySpikeRule()
    result = rule.evaluate(tx, filing_date=filing_date, debtor_id="D001")

    assert result.is_suspicious is True
    assert result.rule_name == "velocity_spike"
    assert result.risk_score > 0
    assert set(result.evidence_ids) == {f"S{i}" for i in range(6)}


def test_velocity_spike_no_spike_for_even_distribution():
    filing_date = pd.Timestamp("2026-06-30")

    # 6개 주 동안 매주 동일하게 2건씩 — "급증"이라 부를 상대적 튐이 없다.
    dates = pd.date_range("2026-05-04", periods=6, freq="W-MON")
    rows = []
    for i, week_start in enumerate(dates):
        rows.append(tx_row(f"E{i}A", week_start, 500_000))
        rows.append(tx_row(f"E{i}B", week_start + pd.Timedelta(days=1), 500_000))

    tx = make_tx(rows)
    rule = VelocitySpikeRule()
    result = rule.evaluate(tx, filing_date=filing_date, debtor_id="D001")

    assert result.is_suspicious is False
    assert result.risk_score == 0


def test_velocity_spike_ignores_non_transfer_transactions():
    filing_date = pd.Timestamp("2026-06-30")

    baseline = [
        tx_row(f"B{i}", date, 200_000)
        for i, date in enumerate([
            "2026-04-13", "2026-04-20", "2026-04-27", "2026-05-04", "2026-05-11",
        ])
    ]
    # 급증처럼 보이는 활동이지만 전부 카드결제 — 송금이 아니므로 대상이 아니다.
    spike = [
        tx_row(f"S{i}", "2026-06-22", 1_000_000, tx_type="카드결제")
        for i in range(6)
    ]

    tx = make_tx(baseline + spike)
    rule = VelocitySpikeRule()
    result = rule.evaluate(tx, filing_date=filing_date, debtor_id="D001")

    assert result.is_suspicious is False


def test_velocity_spike_returns_clean_result_with_single_week():
    filing_date = pd.Timestamp("2026-06-30")

    # 활동 전체가 한 주 안에서만 일어나면 비교할 기준선이 없다.
    tx = make_tx([
        tx_row(f"T{i}", "2026-06-23", 1_000_000) for i in range(6)
    ])
    rule = VelocitySpikeRule()
    result = rule.evaluate(tx, filing_date=filing_date, debtor_id="D001")

    assert result.is_suspicious is False
