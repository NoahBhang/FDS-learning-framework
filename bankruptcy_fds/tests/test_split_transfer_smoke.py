import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트를 import 경로에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bankruptcy_fds.src.rules.split_transfer import SplitTransferRule


def make_tx(rows):
    return pd.DataFrame(rows)


def test_split_transfer_detects_repeated_transfers_to_same_receiver():
    filing_date = pd.Timestamp("2026-06-30")

    tx = make_tx([
        {
            "transaction_id": "T001",
            "debtor_id": "D001",
            "transaction_date": pd.Timestamp("2026-06-01"),
            "amount": 1_000_000,
            "sender": "D001",
            "receiver": "김00",
            "relation_type": "가족",
            "transaction_type": "송금",
        },
        {
            "transaction_id": "T002",
            "debtor_id": "D001",
            "transaction_date": pd.Timestamp("2026-06-03"),
            "amount": 1_000_000,
            "sender": "D001",
            "receiver": "김00",
            "relation_type": "가족",
            "transaction_type": "송금",
        },
        {
            "transaction_id": "T003",
            "debtor_id": "D001",
            "transaction_date": pd.Timestamp("2026-06-05"),
            "amount": 1_000_000,
            "sender": "D001",
            "receiver": "김00",
            "relation_type": "가족",
            "transaction_type": "송금",
        },
        {
            "transaction_id": "T004",
            "debtor_id": "D001",
            "transaction_date": pd.Timestamp("2026-06-07"),
            "amount": 1_000_000,
            "sender": "D001",
            "receiver": "김00",
            "relation_type": "가족",
            "transaction_type": "송금",
        },
        {
            "transaction_id": "T005",
            "debtor_id": "D001",
            "transaction_date": pd.Timestamp("2026-06-09"),
            "amount": 1_000_000,
            "sender": "D001",
            "receiver": "김00",
            "relation_type": "가족",
            "transaction_type": "송금",
        },
    ])

    rule = SplitTransferRule()
    result = rule.evaluate(tx, filing_date=filing_date, debtor_id="D001")

    assert result.is_suspicious is True
    assert result.rule_name == "split_transfer"
    assert result.risk_score > 0
    assert len(result.evidence_ids) == 5


def test_split_transfer_ignores_non_transfer_transactions():
    filing_date = pd.Timestamp("2026-06-30")

    tx = make_tx([
        {
            "transaction_id": f"T{i:03d}",
            "debtor_id": "D001",
            "transaction_date": pd.Timestamp("2026-06-01"),
            "amount": 1_000_000,
            "sender": "D001",
            "receiver": "김00",
            "relation_type": "가족",
            "transaction_type": "카드결제",
        }
        for i in range(1, 6)
    ])

    rule = SplitTransferRule()
    result = rule.evaluate(tx, filing_date=filing_date, debtor_id="D001")

    assert result.is_suspicious is False
    assert result.risk_score == 0
