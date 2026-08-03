"""Compatibility facade tests for the Plugin Rule predictor."""

from datetime import datetime
import inspect

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.models import plugin_rule_predictor as facade_module
from kaggle_bank_fds.src.models.plugin_rule_predictor import predict_with_plugins
from kaggle_bank_fds.src.models.predictor import predict as legacy_predict
from kaggle_bank_fds.src.rules.bank_fraud_rules import (
    FullBalanceTransferRule as LegacyFullBalanceTransferRule,
)
from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.evidence_item import EvidenceItem
from kaggle_bank_fds.src.rules.large_transfer_rule import LargeTransferRule
from kaggle_bank_fds.src.rules.rule_engine import RuleEngine
from kaggle_bank_fds.src.rules.rule_result import RuleResult


def _row(
    *,
    step: int,
    action: str,
    amount: float,
    actor: str,
    target: str,
    balance: float = 2000.0,
) -> dict:
    return {
        "step": step,
        "type": action,
        "amount": amount,
        "nameOrig": actor,
        "oldbalanceOrg": balance,
        "newbalanceOrig": balance - amount,
        "nameDest": target,
        "oldbalanceDest": 0.0,
        "newbalanceDest": amount,
    }


def _normal(index=None) -> pd.DataFrame:
    return pd.DataFrame(
        [_row(step=1, action="PAYMENT", amount=100.0, actor="A", target="M")],
        index=index,
    )


def _full_balance(index=None) -> pd.DataFrame:
    return pd.DataFrame(
        [_row(step=1, action="TRANSFER", amount=1000.0, actor="A", target="B", balance=1000.0)],
        index=index,
    )


def _pair(index=None, cashout_count: int = 1, full_balance: bool = False) -> pd.DataFrame:
    transfer_balance = 1000.0 if full_balance else 2000.0
    rows = [
        _row(
            step=1,
            action="TRANSFER",
            amount=1000.0,
            actor="SOURCE",
            target="MIDDLE",
            balance=transfer_balance,
        )
    ]
    rows.extend(
        _row(
            step=2 + position,
            action="CASH_OUT",
            amount=1000.0,
            actor="MIDDLE",
            target=f"SINK-{position}",
            balance=2000.0,
        )
        for position in range(cashout_count)
    )
    return pd.DataFrame(rows, index=index)


class CleanRule(BaseRule):
    rule_id = "clean"
    rule_name = "Clean Rule"
    description = "Custom clean risk description."

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=False,
            score=0,
            reason="No custom match.",
        )


class EvidenceRule(BaseRule):
    rule_id = "evidence"
    rule_name = "Evidence Rule"
    description = "Custom evidence risk description."
    source_ids = (0,)
    score = 60

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        evidence = tuple(
            EvidenceItem(
                transaction_id=f"custom-{position}",
                source_row_id=source_id,
                actor_account=None,
                target_account=None,
                transaction_datetime=None,
                amount=1.0,
                message="Custom evidence.",
            )
            for position, source_id in enumerate(self.source_ids)
        )
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            triggered=True,
            score=self.score,
            reason="Custom match.",
            evidence=evidence,
        )


class FailureRule(BaseRule):
    rule_id = "failure"
    rule_name = "Failure Rule"
    description = "Always fails."

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        raise RuntimeError("facade boom")


def test_public_import_and_signature() -> None:
    signature = inspect.signature(predict_with_plugins)
    assert list(signature.parameters) == ["transaction_data", "rules"]
    assert signature.parameters["rules"].kind is inspect.Parameter.KEYWORD_ONLY


def test_rules_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        predict_with_plugins(_normal(), [])


@pytest.mark.parametrize("value", [None, [], {}, "frame"])
def test_non_dataframe_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        predict_with_plugins(value)


def test_empty_dataframe_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        predict_with_plugins(pd.DataFrame())


def test_exact_top_level_contract_and_order() -> None:
    result = predict_with_plugins(_normal())
    assert list(result) == ["fraud_score", "triggered_rules", "details"]
    assert set(result) == {"fraud_score", "triggered_rules", "details"}


def test_default_rules_order_and_large_transfer_exclusion() -> None:
    result = predict_with_plugins(_normal())
    assert list(result["details"]) == [
        "transfer_cash_out",
        "full_balance_transfer",
        "skipped_rules",
    ]
    assert "large_transfer" not in result["details"]


def test_default_rules_are_fresh_per_call(monkeypatch) -> None:
    captured = []
    original = RuleEngine.evaluate

    def capture(self, transactions):
        captured.append(tuple(self.registry.get_rules()))
        return original(self, transactions)

    monkeypatch.setattr(RuleEngine, "evaluate", capture)
    predict_with_plugins(_normal())
    predict_with_plugins(_normal())
    assert all(first is not second for first, second in zip(captured[0], captured[1]))


def test_clean_default_result() -> None:
    result = predict_with_plugins(_normal())
    assert result["fraud_score"] == 0.0
    assert type(result["fraud_score"]) is float
    assert result["triggered_rules"] == []
    assert result["details"]["skipped_rules"] == {}
    for rule_id in ("transfer_cash_out", "full_balance_transfer"):
        assert result["details"][rule_id]["is_suspicious"] is False
        assert result["details"][rule_id]["risk_score"] == 0
        assert result["details"][rule_id]["evidence_ids"] == []


def test_single_transfer_cashout_result() -> None:
    result = predict_with_plugins(_pair(index=["transfer", "cashout"]))
    assert result["fraud_score"] == 0.25
    assert result["triggered_rules"] == ["transfer_cash_out"]
    detail = result["details"]["transfer_cash_out"]
    assert detail["risk_type"] == "이체 후 즉시 현금화 의심"
    assert detail["risk_score"] == 25
    assert detail["is_suspicious"] is True
    assert detail["reason"]
    assert detail["evidence_ids"] == ["transfer", "cashout"]


def test_single_full_balance_result() -> None:
    result = predict_with_plugins(_full_balance(index=["full"]))
    assert result["fraud_score"] == 0.2
    assert result["triggered_rules"] == ["full_balance_transfer"]
    detail = result["details"]["full_balance_transfer"]
    assert detail["risk_type"] == "계좌 전액 이체 의심"
    assert detail["risk_score"] == 20
    assert detail["evidence_ids"] == ["full"]


def test_both_default_rules_and_legacy_score_equivalence() -> None:
    frame = _pair(index=["transfer", "cashout"], full_balance=True)
    plugin = predict_with_plugins(frame)
    legacy = legacy_predict(frame)
    assert plugin["fraud_score"] == legacy["fraud_score"] == 0.45
    assert plugin["triggered_rules"] == legacy["triggered_rules"] == [
        "transfer_cash_out",
        "full_balance_transfer",
    ]
    for rule_id in plugin["triggered_rules"]:
        assert plugin["details"][rule_id]["risk_score"] == legacy["details"][rule_id]["risk_score"]
        assert plugin["details"][rule_id]["evidence_ids"] == legacy["details"][rule_id]["evidence_ids"]


def test_default_risk_types_match_legacy_public_contract() -> None:
    frame = _normal()
    legacy = legacy_predict(frame)
    plugin = predict_with_plugins(frame)

    assert (
        legacy["details"]["transfer_cash_out"]["risk_type"]
        == "이체 후 즉시 현금화 의심"
    )
    assert (
        plugin["details"]["transfer_cash_out"]["risk_type"]
        == "이체 후 즉시 현금화 의심"
    )
    assert (
        legacy["details"]["transfer_cash_out"]["risk_type"]
        == plugin["details"]["transfer_cash_out"]["risk_type"]
    )
    assert (
        legacy["details"]["full_balance_transfer"]["risk_type"]
        == plugin["details"]["full_balance_transfer"]["risk_type"]
        == "계좌 전액 이체 의심"
    )


@pytest.mark.parametrize("cashout_count,score", [(2, 0.3), (3, 0.35), (4, 0.4), (8, 0.4)])
def test_transfer_pair_score_and_cap(cashout_count: int, score: float) -> None:
    result = predict_with_plugins(_pair(cashout_count=cashout_count))
    assert result["fraud_score"] == score


def test_total_score_is_capped_at_one() -> None:
    class SecondEvidenceRule(EvidenceRule):
        rule_id = "second-evidence"
        rule_name = "Second Evidence Rule"

    result = predict_with_plugins(_normal(), rules=[EvidenceRule(), SecondEvidenceRule()])
    assert result["fraud_score"] == 1.0


def test_custom_plugin_rules_preserve_order_and_description() -> None:
    result = predict_with_plugins(_normal(), rules=[CleanRule(), EvidenceRule()])
    assert list(result["details"]) == ["clean", "evidence", "skipped_rules"]
    assert result["triggered_rules"] == ["evidence"]
    assert result["details"]["clean"]["risk_type"] == CleanRule.description
    assert result["details"]["evidence"]["risk_type"] == EvidenceRule.description


@pytest.mark.parametrize("rules", [(), (CleanRule(),), {CleanRule()}, "rules"])
def test_rules_must_be_a_list(rules: object) -> None:
    with pytest.raises(TypeError, match="list"):
        predict_with_plugins(_normal(), rules=rules)


@pytest.mark.parametrize("item", [object(), True, LegacyFullBalanceTransferRule()])
def test_non_plugin_and_legacy_rules_are_rejected(item: object) -> None:
    with pytest.raises(TypeError, match="BaseRule"):
        predict_with_plugins(_normal(), rules=[item])


def test_mixed_legacy_and_plugin_rules_are_rejected() -> None:
    with pytest.raises(TypeError):
        predict_with_plugins(_normal(), rules=[CleanRule(), LegacyFullBalanceTransferRule()])


def test_duplicate_rule_id_is_registry_error() -> None:
    with pytest.raises(ValueError, match="rule_id"):
        predict_with_plugins(_normal(), rules=[CleanRule(), CleanRule()])


def test_custom_rules_list_is_not_mutated() -> None:
    rules = [CleanRule(), EvidenceRule()]
    before = list(rules)
    predict_with_plugins(_normal(), rules=rules)
    assert rules == before
    assert all(actual is expected for actual, expected in zip(rules, before))


@pytest.mark.parametrize(
    "index,expected",
    [
        (None, [0]),
        ([101], [101]),
        (["raw-row"], ["raw-row"]),
    ],
)
def test_evidence_restores_raw_index(index, expected) -> None:
    result = predict_with_plugins(_full_balance(index=index))
    assert result["details"]["full_balance_transfer"]["evidence_ids"] == expected


def test_noncontiguous_and_string_pair_indexes_are_restored() -> None:
    result = predict_with_plugins(_pair(index=[100, "cashout-z"]))
    assert result["details"]["transfer_cash_out"]["evidence_ids"] == [100, "cashout-z"]


def test_duplicate_raw_indexes_are_first_seen_deduplicated() -> None:
    result = predict_with_plugins(_pair(index=["same", "same"]))
    assert result["details"]["transfer_cash_out"]["evidence_ids"] == ["same"]


def test_repeated_pair_evidence_is_flattened_and_deduplicated() -> None:
    result = predict_with_plugins(_pair(index=["transfer", "cash-1", "cash-2"], cashout_count=2))
    assert result["details"]["transfer_cash_out"]["evidence_ids"] == [
        "transfer",
        "cash-1",
        "cash-2",
    ]


def test_none_source_row_id_is_omitted() -> None:
    class NoneEvidenceRule(EvidenceRule):
        source_ids = (None,)

    result = predict_with_plugins(_normal(), rules=[NoneEvidenceRule()])
    assert result["details"]["evidence"]["evidence_ids"] == []


@pytest.mark.parametrize("source_id", ["0", 9, -1])
def test_invalid_evidence_source_row_id_is_mapping_error(source_id: object) -> None:
    class InvalidEvidenceRule(EvidenceRule):
        source_ids = (source_id,)

    with pytest.raises((TypeError, ValueError), match="source_row_id"):
        predict_with_plugins(_normal(), rules=[InvalidEvidenceRule()])


def test_rule_failure_is_skipped_and_next_rule_runs() -> None:
    result = predict_with_plugins(_normal(), rules=[FailureRule(), EvidenceRule()])
    assert result["triggered_rules"] == ["evidence"]
    assert result["details"]["skipped_rules"] == {"failure": "facade boom"}
    assert "RuntimeError" not in result["details"]["skipped_rules"]["failure"]


def test_adapter_error_is_not_skipped() -> None:
    with pytest.raises(ValueError, match="amount"):
        predict_with_plugins(_normal().assign(amount="invalid"))


def test_normal_path_does_not_mutate_input() -> None:
    frame = _pair(index=["transfer", "cashout"], full_balance=True)
    before = frame.copy(deep=True)
    predict_with_plugins(frame)
    assert_frame_equal(frame, before)


def test_rule_failure_path_does_not_mutate_input() -> None:
    frame = _normal(index=["normal"])
    before = frame.copy(deep=True)
    predict_with_plugins(frame, rules=[FailureRule(), CleanRule()])
    assert_frame_equal(frame, before)


def test_adapter_failure_path_does_not_mutate_input() -> None:
    frame = _normal().assign(amount="invalid")
    before = frame.copy(deep=True)
    with pytest.raises(ValueError):
        predict_with_plugins(frame)
    assert_frame_equal(frame, before)


def test_full_balance_default_reason_remains_plugin_reason() -> None:
    result = predict_with_plugins(_full_balance())
    reason = result["details"]["full_balance_transfer"]["reason"]
    assert reason.startswith("Detected 1 full or effectively full balance transfer")
    assert "계좌 잔액 전액" not in reason


def test_large_transfer_is_supported_only_when_explicitly_configured() -> None:
    result = predict_with_plugins(
        _normal(),
        rules=[LargeTransferRule(threshold=50, score=70)],
    )
    assert result["triggered_rules"] == ["large_transfer"]
    assert result["fraud_score"] == 0.7
    assert result["details"]["large_transfer"]["risk_type"] == LargeTransferRule.description
