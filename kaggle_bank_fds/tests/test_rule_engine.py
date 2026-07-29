"""Sprint 2.0.1 Plugin Rule 계약 단위 테스트."""

from dataclasses import FrozenInstanceError
from datetime import datetime
import inspect
import math

import pandas as pd
import pytest

from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.dummy_rules import (
    AlwaysFalseRule,
    AlwaysTrueRule,
)
from kaggle_bank_fds.src.rules.evidence_item import EvidenceItem
from kaggle_bank_fds.src.rules.rule_engine import RuleEngine
from kaggle_bank_fds.src.rules.rule_engine_report import RuleEngineReport
from kaggle_bank_fds.src.rules.rule_execution_error import (
    RuleExecutionError,
)
from kaggle_bank_fds.src.rules.rule_registry import RuleRegistry
from kaggle_bank_fds.src.rules.rule_result import RuleResult


def make_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["TX-1", "TX-2"],
            "source_row_id": pd.Series([0, 1], dtype="Int64"),
            "transaction_datetime": pd.to_datetime(
                ["2026-07-01 09:00", "2026-07-01 10:00"]
            ),
            "amount": pd.Series([10_000, 20_000], dtype="Float64"),
            "actor_account": pd.Series(
                ["ACCOUNT-A", "ACCOUNT-B"],
                dtype="string",
            ),
            "target_account": pd.Series(
                ["ACCOUNT-B", "ACCOUNT-C"],
                dtype="string",
            ),
        }
    )


def make_evidence(
    transaction_id: str = "TX-1",
) -> EvidenceItem:
    return EvidenceItem(
        transaction_id=transaction_id,
        source_row_id=0,
        actor_account="ACCOUNT-A",
        target_account="ACCOUNT-B",
        transaction_datetime=datetime(2026, 7, 1, 9, 0),
        amount=10_000,
        message="테스트 근거",
    )


class BrokenRule(BaseRule):
    rule_id = "broken"
    rule_name = "Broken Rule"
    description = "RuntimeError 격리 테스트"

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        raise RuntimeError("boom")


class WrongReturnRule(BaseRule):
    rule_id = "wrong_return"
    rule_name = "Wrong Return Rule"
    description = "잘못된 반환 타입 테스트"

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        return "not a RuleResult"


class KeyboardInterruptRule(BaseRule):
    rule_id = "keyboard_interrupt"
    rule_name = "Keyboard Interrupt Rule"
    description = "BaseException 전파 테스트"

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        raise KeyboardInterrupt


class SystemExitRule(BaseRule):
    rule_id = "system_exit"
    rule_name = "System Exit Rule"
    description = "SystemExit 전파 테스트"

    def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
        raise SystemExit(2)


def test_evidence_item_normal_creation() -> None:
    evidence = make_evidence()

    assert evidence.transaction_id == "TX-1"
    assert evidence.amount == 10_000.0


def test_evidence_item_strips_string_fields() -> None:
    evidence = EvidenceItem(
        transaction_id=" TX-1 ",
        source_row_id=" row-1 ",
        actor_account=" A ",
        target_account=" B ",
        transaction_datetime=None,
        amount=None,
        message=" 근거 ",
    )

    assert evidence.transaction_id == "TX-1"
    assert evidence.source_row_id == "row-1"
    assert evidence.actor_account == "A"
    assert evidence.target_account == "B"
    assert evidence.message == "근거"


@pytest.mark.parametrize("transaction_id", ["", " ", "\t"])
def test_evidence_item_rejects_blank_transaction_id(
    transaction_id,
) -> None:
    with pytest.raises(ValueError, match="transaction_id"):
        EvidenceItem(
            transaction_id=transaction_id,
            source_row_id=None,
            actor_account=None,
            target_account=None,
            transaction_datetime=None,
            amount=None,
            message="근거",
        )


def test_evidence_item_rejects_blank_message() -> None:
    with pytest.raises(ValueError, match="message"):
        EvidenceItem(
            transaction_id="TX-1",
            source_row_id=None,
            actor_account=None,
            target_account=None,
            transaction_datetime=None,
            amount=None,
            message=" ",
        )


@pytest.mark.parametrize("amount", [math.nan, math.inf, -math.inf])
def test_evidence_item_rejects_non_finite_amount(amount) -> None:
    with pytest.raises(ValueError, match="유한한 숫자"):
        EvidenceItem(
            transaction_id="TX-1",
            source_row_id=None,
            actor_account=None,
            target_account=None,
            transaction_datetime=None,
            amount=amount,
            message="근거",
        )


def test_evidence_item_rejects_invalid_source_row_id() -> None:
    with pytest.raises(TypeError, match="source_row_id"):
        EvidenceItem(
            transaction_id="TX-1",
            source_row_id=1.5,
            actor_account=None,
            target_account=None,
            transaction_datetime=None,
            amount=None,
            message="근거",
        )


def test_evidence_item_rejects_invalid_datetime() -> None:
    with pytest.raises(TypeError, match="transaction_datetime"):
        EvidenceItem(
            transaction_id="TX-1",
            source_row_id=None,
            actor_account=None,
            target_account=None,
            transaction_datetime="2026-07-01",
            amount=None,
            message="근거",
        )


def test_evidence_item_is_frozen() -> None:
    evidence = make_evidence()

    with pytest.raises(FrozenInstanceError):
        evidence.message = "변경"


def test_triggered_rule_result_is_valid() -> None:
    result = RuleResult(
        rule_id="rule",
        rule_name="Rule",
        triggered=True,
        score=80,
        reason="탐지",
        evidence=(make_evidence(),),
    )

    assert result.score == 80


def test_non_triggered_rule_result_is_valid() -> None:
    result = RuleResult(
        rule_id="rule",
        rule_name="Rule",
        triggered=False,
        score=0,
        reason="미탐지",
    )

    assert result.evidence == ()


@pytest.mark.parametrize("score", [-1, 101])
def test_rule_result_rejects_out_of_range_score(score) -> None:
    with pytest.raises(ValueError, match="0 이상 100 이하"):
        RuleResult(
            rule_id="rule",
            rule_name="Rule",
            triggered=True,
            score=score,
            reason="탐지",
            evidence=(make_evidence(),),
        )


@pytest.mark.parametrize("score", [1.0, True])
def test_rule_result_rejects_non_integer_score(score) -> None:
    with pytest.raises(TypeError, match="bool이 아닌 int"):
        RuleResult(
            rule_id="rule",
            rule_name="Rule",
            triggered=True,
            score=score,
            reason="탐지",
            evidence=(make_evidence(),),
        )


def test_rule_result_rejects_false_with_positive_score() -> None:
    with pytest.raises(ValueError, match="score는 0"):
        RuleResult(
            rule_id="rule",
            rule_name="Rule",
            triggered=False,
            score=1,
            reason="미탐지",
        )


def test_rule_result_rejects_false_with_evidence() -> None:
    with pytest.raises(ValueError, match="evidence는 비어"):
        RuleResult(
            rule_id="rule",
            rule_name="Rule",
            triggered=False,
            score=0,
            reason="미탐지",
            evidence=(make_evidence(),),
        )


def test_rule_result_rejects_true_with_zero_score() -> None:
    with pytest.raises(ValueError, match="score는 1 이상"):
        RuleResult(
            rule_id="rule",
            rule_name="Rule",
            triggered=True,
            score=0,
            reason="탐지",
            evidence=(make_evidence(),),
        )


def test_rule_result_rejects_true_without_evidence() -> None:
    with pytest.raises(ValueError, match="evidence가 필요"):
        RuleResult(
            rule_id="rule",
            rule_name="Rule",
            triggered=True,
            score=50,
            reason="탐지",
        )


def test_rule_result_normalizes_evidence_list_to_tuple() -> None:
    evidence = make_evidence()
    result = RuleResult(
        rule_id="rule",
        rule_name="Rule",
        triggered=True,
        score=50,
        reason="탐지",
        evidence=[evidence],
    )

    assert result.evidence == (evidence,)


def test_rule_result_evidence_is_immutable() -> None:
    result = RuleResult(
        rule_id="rule",
        rule_name="Rule",
        triggered=True,
        score=50,
        reason="탐지",
        evidence=(make_evidence(),),
    )

    with pytest.raises(AttributeError):
        result.evidence.append(make_evidence("TX-2"))


def test_rule_result_is_frozen() -> None:
    result = RuleResult(
        rule_id="rule",
        rule_name="Rule",
        triggered=False,
        score=0,
        reason="미탐지",
    )

    with pytest.raises(FrozenInstanceError):
        result.score = 10


def test_valid_concrete_rule_metadata() -> None:
    class ValidRule(BaseRule):
        rule_id = "valid"
        rule_name = "Valid Rule"
        description = "정상 Rule"

        def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
            return AlwaysFalseRule().evaluate(transactions)

    assert ValidRule.rule_id == "valid"


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("rule_id", " "),
        ("rule_name", ""),
        ("description", "\t"),
    ],
)
def test_concrete_rule_rejects_blank_metadata(
    field_name,
    field_value,
) -> None:
    metadata = {
        "rule_id": "valid",
        "rule_name": "Valid Rule",
        "description": "정상 Rule",
        field_name: field_value,
        "evaluate": lambda self, transactions: None,
    }

    with pytest.raises(ValueError, match=field_name):
        type("InvalidRule", (BaseRule,), metadata)


def test_concrete_rule_rejects_missing_metadata() -> None:
    with pytest.raises(TypeError, match="rule_id"):
        type(
            "MissingMetadataRule",
            (BaseRule,),
            {"evaluate": lambda self, transactions: None},
        )


def test_rule_without_evaluate_remains_abstract() -> None:
    class AbstractRule(BaseRule):
        pass

    assert inspect.isabstract(AbstractRule)


def test_registry_registers_and_gets_rule() -> None:
    registry = RuleRegistry()
    rule = AlwaysTrueRule()

    registry.register(rule)

    assert registry.get(rule.rule_id) is rule


def test_registry_preserves_registration_order() -> None:
    registry = RuleRegistry()
    true_rule = AlwaysTrueRule()
    false_rule = AlwaysFalseRule()
    registry.register(true_rule)
    registry.register(false_rule)

    assert registry.get_rules() == [true_rule, false_rule]


def test_registry_rejects_duplicate_rule_id() -> None:
    registry = RuleRegistry()
    registry.register(AlwaysTrueRule())

    with pytest.raises(ValueError, match="이미 등록된"):
        registry.register(AlwaysTrueRule())


def test_registry_rejects_non_rule() -> None:
    with pytest.raises(TypeError, match="BaseRule"):
        RuleRegistry().register(object())


def test_registry_rejects_mutated_invalid_metadata() -> None:
    rule = AlwaysTrueRule()
    rule.rule_id = " always_true "

    with pytest.raises(ValueError, match="앞뒤"):
        RuleRegistry().register(rule)


def test_registry_returns_copy() -> None:
    registry = RuleRegistry()
    registry.register(AlwaysTrueRule())

    returned = registry.get_rules()
    returned.clear()

    assert len(registry.get_rules()) == 1


def test_registry_unknown_id_raises_key_error() -> None:
    with pytest.raises(KeyError, match="등록되지 않은"):
        RuleRegistry().get("missing")


def test_engine_executes_all_rules_in_order() -> None:
    registry = RuleRegistry()
    registry.register(AlwaysTrueRule())
    registry.register(AlwaysFalseRule())

    report = RuleEngine(registry).evaluate(make_transactions())

    assert [result.rule_id for result in report.results] == [
        "always_true",
        "always_false",
    ]


def test_engine_empty_registry_returns_empty_report() -> None:
    report = RuleEngine(RuleRegistry()).evaluate(make_transactions())

    assert report == RuleEngineReport()


def test_engine_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        RuleEngine(RuleRegistry()).evaluate([])


def test_engine_continues_after_runtime_error() -> None:
    registry = RuleRegistry()
    registry.register(BrokenRule())
    registry.register(AlwaysFalseRule())

    report = RuleEngine(registry).evaluate(make_transactions())

    assert [result.rule_id for result in report.results] == [
        "always_false"
    ]
    assert [error.rule_id for error in report.errors] == ["broken"]


def test_engine_records_error_type_and_message() -> None:
    registry = RuleRegistry()
    registry.register(BrokenRule())

    error = RuleEngine(registry).evaluate(
        make_transactions()
    ).errors[0]

    assert error.error_type == "RuntimeError"
    assert error.message == "boom"


def test_engine_records_invalid_return_type_as_error() -> None:
    registry = RuleRegistry()
    registry.register(WrongReturnRule())

    error = RuleEngine(registry).evaluate(
        make_transactions()
    ).errors[0]

    assert error.error_type == "TypeError"
    assert "RuleResult" in error.message


def test_engine_report_counts_and_triggered_results() -> None:
    registry = RuleRegistry()
    registry.register(AlwaysTrueRule())
    registry.register(BrokenRule())
    registry.register(AlwaysFalseRule())

    report = RuleEngine(registry).evaluate(make_transactions())

    assert report.succeeded_count == 2
    assert report.failed_count == 1
    assert [item.rule_id for item in report.triggered_results] == [
        "always_true"
    ]


def test_engine_does_not_swallow_keyboard_interrupt() -> None:
    registry = RuleRegistry()
    registry.register(KeyboardInterruptRule())

    with pytest.raises(KeyboardInterrupt):
        RuleEngine(registry).evaluate(make_transactions())


def test_engine_does_not_swallow_system_exit() -> None:
    registry = RuleRegistry()
    registry.register(SystemExitRule())

    with pytest.raises(SystemExit):
        RuleEngine(registry).evaluate(make_transactions())


def test_always_true_uses_canonical_transaction_ids() -> None:
    result = AlwaysTrueRule().evaluate(make_transactions())

    assert result.triggered is True
    assert result.score == 100
    assert [item.transaction_id for item in result.evidence] == [
        "TX-1",
        "TX-2",
    ]
    assert [item.source_row_id for item in result.evidence] == [0, 1]


def test_always_false_returns_valid_clean_result() -> None:
    result = AlwaysFalseRule().evaluate(make_transactions())

    assert result.triggered is False
    assert result.score == 0
    assert result.evidence == ()


def test_always_true_requires_transaction_id_column() -> None:
    transactions = make_transactions().drop(
        columns=["transaction_id"]
    )

    with pytest.raises(ValueError, match="transaction_id 컬럼"):
        AlwaysTrueRule().evaluate(transactions)


def test_rule_engine_report_is_frozen_and_uses_tuples() -> None:
    report = RuleEngineReport(results=[], errors=[])

    assert report.results == ()
    assert report.errors == ()
    with pytest.raises(FrozenInstanceError):
        report.results = ()


def test_rule_execution_error_is_frozen() -> None:
    error = RuleExecutionError(
        rule_id=" rule ",
        rule_name=" Rule ",
        error_type=" RuntimeError ",
        message=" boom ",
    )

    assert error.rule_id == "rule"
    assert error.message == "boom"
    with pytest.raises(FrozenInstanceError):
        error.message = "changed"


def test_evidence_item_rejects_boolean_source_row_id() -> None:
    with pytest.raises(TypeError, match="source_row_id"):
        EvidenceItem(
            transaction_id="TX-1",
            source_row_id=True,
            actor_account=None,
            target_account=None,
            transaction_datetime=None,
            amount=None,
            message="근거",
        )


def test_evidence_item_rejects_boolean_amount() -> None:
    with pytest.raises(TypeError, match="amount"):
        EvidenceItem(
            transaction_id="TX-1",
            source_row_id=0,
            actor_account=None,
            target_account=None,
            transaction_datetime=None,
            amount=True,
            message="근거",
        )


def test_evidence_item_normalizes_nat_to_none() -> None:
    evidence = EvidenceItem(
        transaction_id="TX-1",
        source_row_id=0,
        actor_account=None,
        target_account=None,
        transaction_datetime=pd.NaT,
        amount=None,
        message="근거",
    )

    assert evidence.transaction_datetime is None


@pytest.mark.parametrize("triggered", [1, 0])
def test_rule_result_rejects_integer_triggered(triggered) -> None:
    with pytest.raises(TypeError, match="triggered는 bool"):
        RuleResult(
            rule_id="rule",
            rule_name="Rule",
            triggered=triggered,
            score=0,
            reason="결과",
        )


def test_rule_result_rejects_non_evidence_item() -> None:
    with pytest.raises(TypeError, match="EvidenceItem"):
        RuleResult(
            rule_id="rule",
            rule_name="Rule",
            triggered=True,
            score=50,
            reason="탐지",
            evidence=("TX-1",),
        )


def test_rule_engine_report_rejects_non_rule_result() -> None:
    with pytest.raises(TypeError, match="RuleResult"):
        RuleEngineReport(results=["invalid"])


def test_rule_engine_report_rejects_non_execution_error() -> None:
    with pytest.raises(TypeError, match="RuleExecutionError"):
        RuleEngineReport(errors=["invalid"])


def test_dummy_rule_normalizes_nullable_accounts_to_none() -> None:
    transactions = make_transactions()
    transactions.loc[0, "actor_account"] = pd.NA
    transactions.loc[1, "target_account"] = pd.NA

    evidence = AlwaysTrueRule().evaluate(transactions).evidence

    assert evidence[0].actor_account is None
    assert evidence[1].target_account is None
    assert evidence[0].actor_account not in {"<NA>", "nan"}
    assert evidence[1].target_account not in {"<NA>", "nan"}


def test_dummy_rule_normalizes_nat_to_none() -> None:
    transactions = make_transactions()
    transactions.loc[0, "transaction_datetime"] = pd.NaT

    evidence = AlwaysTrueRule().evaluate(transactions).evidence

    assert evidence[0].transaction_datetime is None


def test_dummy_rule_normalizes_nullable_amount_to_none() -> None:
    transactions = make_transactions()
    transactions.loc[0, "amount"] = pd.NA

    evidence = AlwaysTrueRule().evaluate(transactions).evidence

    assert evidence[0].amount is None


def test_engine_preserves_success_order_across_failure() -> None:
    executed: list[str] = []

    class FirstSuccessRule(BaseRule):
        rule_id = "first_success"
        rule_name = "First Success"
        description = "첫 번째 성공 Rule"

        def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
            executed.append(self.rule_id)
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=False,
                score=0,
                reason="정상 실행",
            )

    class MiddleFailureRule(BaseRule):
        rule_id = "middle_failure"
        rule_name = "Middle Failure"
        description = "중간 실패 Rule"

        def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
            executed.append(self.rule_id)
            raise RuntimeError("middle failure")

    class LastSuccessRule(BaseRule):
        rule_id = "last_success"
        rule_name = "Last Success"
        description = "마지막 성공 Rule"

        def evaluate(self, transactions: pd.DataFrame) -> RuleResult:
            executed.append(self.rule_id)
            return RuleResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                triggered=False,
                score=0,
                reason="정상 실행",
            )

    registry = RuleRegistry()
    registry.register(FirstSuccessRule())
    registry.register(MiddleFailureRule())
    registry.register(LastSuccessRule())

    report = RuleEngine(registry).evaluate(make_transactions())

    assert report.succeeded_count == 2
    assert report.failed_count == 1
    assert [result.rule_id for result in report.results] == [
        "first_success",
        "last_success",
    ]
    assert [error.rule_id for error in report.errors] == [
        "middle_failure"
    ]
    assert executed == [
        "first_success",
        "middle_failure",
        "last_success",
    ]


def test_engine_continues_after_invalid_return_type() -> None:
    registry = RuleRegistry()
    registry.register(WrongReturnRule())
    registry.register(AlwaysFalseRule())

    report = RuleEngine(registry).evaluate(make_transactions())

    assert report.succeeded_count == 1
    assert report.failed_count == 1
    assert report.results[0].rule_id == "always_false"
    assert report.errors[0].rule_id == "wrong_return"
    assert report.errors[0].error_type == "TypeError"
