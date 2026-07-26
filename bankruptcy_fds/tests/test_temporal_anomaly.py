import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bankruptcy_fds.src.rules.temporal_anomaly import TemporalAnomalyRule


def make_tx(rows):
    return pd.DataFrame(rows)


def tx_row(tx_id, date, amount=500_000, receiver="김00", tx_type="송금"):
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


def test_temporal_anomaly_detects_weekend_concentration():
    filing_date = pd.Timestamp("2026-06-30")

    # 날짜만 있고 시각 정보가 없는(자정으로 채워지는) 전형적인 DATE 스키마 값.
    # 토(5)/일(6) 5건 + 평일 1건 → 주말 비율 5/6 ≈ 83%.
    weekend_dates = ["2026-06-06", "2026-06-07", "2026-06-13",
                     "2026-06-14", "2026-06-20"]   # 모두 토/일
    weekday_dates = ["2026-06-09"]                 # 화요일

    rows = [tx_row(f"W{i}", d) for i, d in enumerate(weekend_dates)]
    rows += [tx_row("D0", weekday_dates[0])]

    tx = make_tx(rows)
    rule = TemporalAnomalyRule()
    result = rule.evaluate(tx, filing_date=filing_date, debtor_id="D001")

    assert result.is_suspicious is True
    assert result.rule_name == "temporal_anomaly"
    assert "주말" in result.reason
    assert result.risk_score > 0
    assert set(result.evidence_ids) == {f"W{i}" for i in range(5)}


def test_temporal_anomaly_detects_night_concentration():
    filing_date = pd.Timestamp("2026-06-30")

    # 시각 정보가 실제로 존재하고(hour가 다양함) 야간(00~06시)에 집중된 경우.
    # 모두 평일이라 주말 신호는 걸리지 않는다.
    night_times = [
        "2026-06-01 02:00:00", "2026-06-02 03:00:00",
        "2026-06-03 01:00:00", "2026-06-04 04:00:00",
    ]
    day_times = ["2026-06-05 14:00:00", "2026-06-08 15:00:00"]

    rows = [tx_row(f"N{i}", t) for i, t in enumerate(night_times)]
    rows += [tx_row(f"Y{i}", t) for i, t in enumerate(day_times)]

    tx = make_tx(rows)
    rule = TemporalAnomalyRule()
    result = rule.evaluate(tx, filing_date=filing_date, debtor_id="D001")

    assert result.is_suspicious is True
    assert "야간" in result.reason
    assert set(result.evidence_ids) == {f"N{i}" for i in range(4)}


def test_temporal_anomaly_no_time_resolution_does_not_false_positive_on_night():
    """
    DATE 전용 스키마(시각 없음)에서는 모든 거래의 hour가 0으로 동일해진다.
    이때 0시가 NIGHT_HOURS(0~5)에 속한다는 이유만으로 "야간 집중"을
    오탐해서는 안 된다 — has_time_resolution 가드의 회귀 테스트.
    """
    filing_date = pd.Timestamp("2026-06-30")

    # 전부 평일, 날짜만 있고 시각 없음(hour == 0 for all).
    weekday_dates = ["2026-06-01", "2026-06-02", "2026-06-03",
                     "2026-06-04", "2026-06-05", "2026-06-08"]
    tx = make_tx([tx_row(f"T{i}", d) for i, d in enumerate(weekday_dates)])

    rule = TemporalAnomalyRule()
    result = rule.evaluate(tx, filing_date=filing_date, debtor_id="D001")

    assert result.is_suspicious is False
    assert result.risk_score == 0


def test_temporal_anomaly_ignores_small_sample():
    filing_date = pd.Timestamp("2026-06-30")

    # 전부 주말이지만 표본이 min_count(기본 5)에 못 미친다.
    tx = make_tx([
        tx_row("W0", "2026-06-06"),
        tx_row("W1", "2026-06-07"),
    ])
    rule = TemporalAnomalyRule()
    result = rule.evaluate(tx, filing_date=filing_date, debtor_id="D001")

    assert result.is_suspicious is False
