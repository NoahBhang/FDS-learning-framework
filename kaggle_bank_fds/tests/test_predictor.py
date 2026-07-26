import sys
from pathlib import Path

import pandas as pd
import pytest

# 프로젝트 루트를 import 경로에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kaggle_bank_fds.src.models.predictor import predict, DEFAULT_RULES


def make_tx():
    """이체 후 즉시 현금화 + 전액 이체 패턴을 모두 포함한 배치."""
    return pd.DataFrame([
        {
            "step": 1, "type": "TRANSFER", "amount": 100_000,
            "nameOrig": "C1", "oldbalanceOrg": 100_000, "newbalanceOrig": 0,
            "nameDest": "C2", "oldbalanceDest": 0, "newbalanceDest": 100_000,
        },
        {
            "step": 2, "type": "CASH_OUT", "amount": 99_000,
            "nameOrig": "C2", "oldbalanceOrg": 100_000, "newbalanceOrig": 1_000,
            "nameDest": "C3", "oldbalanceDest": 0, "newbalanceDest": 99_000,
        },
    ])


class BrokenRule:
    """예외 격리 테스트용 — 항상 평가에 실패하는 가짜 규칙."""
    rule_name = "broken_rule"

    def evaluate(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_predict_normal_case_returns_expected_shape():
    result = predict(make_tx())

    assert set(result.keys()) == {"fraud_score", "triggered_rules", "details"}
    assert 0.0 <= result["fraud_score"] <= 1.0
    assert "transfer_cash_out" in result["triggered_rules"]
    assert "full_balance_transfer" in result["triggered_rules"]

    # 의심 판정된 규칙은 details에도 반드시 근거(reason·evidence_ids)를 동반한다.
    for rule_name in result["triggered_rules"]:
        detail = result["details"][rule_name]
        assert detail["is_suspicious"] is True
        assert detail["reason"]
        assert len(detail["evidence_ids"]) > 0

    assert result["details"]["skipped_rules"] == {}


def test_predict_ignores_normal_transactions():
    tx = pd.DataFrame([
        {
            "step": 1, "type": "PAYMENT", "amount": 5_000,
            "nameOrig": "C1", "oldbalanceOrg": 100_000, "newbalanceOrig": 95_000,
            "nameDest": "M1", "oldbalanceDest": 0, "newbalanceDest": 5_000,
        },
    ])
    result = predict(tx)

    assert result["fraud_score"] == 0.0
    assert result["triggered_rules"] == []


def test_predict_isolates_failing_rule_and_records_skipped_rules():
    rules = DEFAULT_RULES + [BrokenRule()]
    result = predict(make_tx(), rules=rules)

    # 실패한 규칙은 triggered_rules에서 빠지고 나머지 규칙은 정상 집계된다.
    assert "broken_rule" not in result["triggered_rules"]
    assert "transfer_cash_out" in result["triggered_rules"]
    assert "full_balance_transfer" in result["triggered_rules"]

    # skipped_rules에 실패 사유가 기록된다.
    assert result["details"]["skipped_rules"] == {"broken_rule": "boom"}

    # 성공한 규칙들의 판단은 실패 규칙이 섞여도 영향받지 않는다.
    baseline = predict(make_tx())
    assert result["fraud_score"] == baseline["fraud_score"]


def test_predict_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        predict(pd.DataFrame())
