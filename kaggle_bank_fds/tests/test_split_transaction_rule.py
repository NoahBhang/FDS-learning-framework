"""Contract tests for SplitTransactionRule."""

from decimal import Decimal
import inspect
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.rule_engine import RuleEngine
from kaggle_bank_fds.src.rules.rule_registry import RuleRegistry
from kaggle_bank_fds.src.rules.split_transaction_rule import SplitTransactionRule


REQUIRED=SplitTransactionRule.REQUIRED_COLUMNS
def _frame(*changes,index=None):
    rows=[]
    for i,c in enumerate(changes or ({},)):
        rows.append({"transaction_id":f"tx-{i}","source_row_id":i,"step":i,
                     "transaction_datetime":pd.Timestamp("2026-01-01"),"action_type":"TRANSFER",
                     "amount":70_000.0,"actor_account":"A","target_account":f"B{i}"}|c)
    return pd.DataFrame(rows,columns=REQUIRED,index=index)


def test_metadata_signature_defaults_readonly():
    rule=SplitTransactionRule();sig=inspect.signature(SplitTransactionRule)
    assert rule.rule_id=="split_transaction"
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values())
    assert (rule.individual_amount_ceiling,rule.cumulative_amount_threshold,rule.max_step_gap,rule.min_count,rule.score)==(100_000.0,200_000.0,24,3,35)
    with pytest.raises(AttributeError): rule.score=1


def test_numpy_parameters_normalized():
    rule=SplitTransactionRule(individual_amount_ceiling=np.float64(10),cumulative_amount_threshold=np.float64(20),max_step_gap=np.int64(1),min_count=np.int64(2),score=np.int64(8))
    assert type(rule.individual_amount_ceiling) is type(rule.cumulative_amount_threshold) is float
    assert all(type(v) is int for v in (rule.max_step_gap,rule.min_count,rule.score))


@pytest.mark.parametrize(("name","value"),[("individual_amount_ceiling",True),("individual_amount_ceiling","1"),("individual_amount_ceiling",Decimal("1")),("cumulative_amount_threshold",None),("cumulative_amount_threshold",1+0j),("max_step_gap",True),("max_step_gap",1.0),("min_count","2"),("score",True),("score",20.0)])
def test_invalid_parameter_types(name,value):
    with pytest.raises(TypeError): SplitTransactionRule(**{name:value})


@pytest.mark.parametrize("kwargs",[{"individual_amount_ceiling":0},{"individual_amount_ceiling":np.inf},{"cumulative_amount_threshold":np.nan},{"max_step_gap":-1},{"min_count":1},{"score":0},{"score":101},{"individual_amount_ceiling":100,"cumulative_amount_threshold":100}])
def test_invalid_parameter_values(kwargs):
    with pytest.raises(ValueError): SplitTransactionRule(**kwargs)


def test_ceiling_and_cumulative_boundaries():
    rule=SplitTransactionRule()
    assert rule.evaluate(_frame({"amount":99_999},{"amount":50_001},{"amount":50_000})).triggered
    assert not rule.evaluate(_frame({"amount":100_000},{"amount":99_999},{"amount":99_999})).triggered
    assert not rule.evaluate(_frame({"amount":70_000},{"amount":70_000},{"amount":59_999})).triggered
    assert rule.evaluate(_frame({"amount":70_000},{"amount":70_000},{"amount":60_000})).triggered


def test_same_sender_multiple_or_same_receivers_allowed():
    assert SplitTransactionRule().evaluate(_frame({}, {}, {})).triggered
    assert SplitTransactionRule().evaluate(_frame({"target_account":"B"},{"target_account":"B"},{"target_account":"B"})).triggered


def test_different_senders_do_not_combine():
    frame=_frame({"actor_account":"A1"},{"actor_account":"A2"},{"actor_account":"A3"})
    assert not SplitTransactionRule().evaluate(frame).triggered


def test_gap_boundary_and_outside():
    assert SplitTransactionRule().evaluate(_frame({"step":0},{"step":24},{"step":24})).triggered
    assert not SplitTransactionRule().evaluate(_frame({"step":0},{"step":25},{"step":25})).triggered


def test_large_single_nonpositive_and_non_candidate_excluded():
    rule=SplitTransactionRule(min_count=2,cumulative_amount_threshold=150_000)
    frame=_frame({"amount":200_000},{"amount":0},{"amount":-1},{"action_type":"PAYMENT","amount":"bad","step":"bad","transaction_id":"","actor_account":None,"target_account":None})
    assert not rule.evaluate(frame).triggered


def test_ceiling_or_higher_transaction_is_excluded_but_does_not_split_window():
    rule = SplitTransactionRule(
        individual_amount_ceiling=100_000,
        cumulative_amount_threshold=200_000,
        max_step_gap=3,
        min_count=3,
    )
    frame = _frame(
        {"step": 1, "amount": 70_000},
        {"step": 2, "amount": 150_000},
        {"step": 3, "amount": 70_000},
        {"step": 4, "amount": 70_000},
    )
    before = frame.copy(deep=True)

    result = rule.evaluate(frame)

    evidence_ids = [item.transaction_id for item in result.evidence]
    assert result.triggered is True
    assert evidence_ids == ["tx-0", "tx-2", "tx-3"]
    assert "tx-1" not in evidence_ids
    assert len(evidence_ids) == len(set(evidence_ids))
    assert result.score == rule.score
    assert_frame_equal(frame, before)


@pytest.mark.parametrize(("field","value","error"),[("amount",pd.NA,ValueError),("amount",np.inf,ValueError),("amount",True,TypeError),("amount","1",TypeError),("step",pd.NA,ValueError),("step",True,TypeError),("step",1.0,TypeError),("actor_account",None,ValueError),("target_account"," ",ValueError),("transaction_id",pd.NA,ValueError)])
def test_invalid_candidate_fields(field,value,error):
    with pytest.raises(error): SplitTransactionRule().evaluate(_frame({field:value}))


def test_duplicate_transaction_rejected_source_allowed():
    with pytest.raises(ValueError,match="unique"): SplitTransactionRule().evaluate(_frame({"transaction_id":"x"},{"transaction_id":"x"}))
    result=SplitTransactionRule(min_count=2,cumulative_amount_threshold=120_000).evaluate(_frame({"source_row_id":7},{"source_row_id":7}))
    assert [e.source_row_id for e in result.evidence]==[7,7]


def test_overlapping_window_union_original_position_and_privacy():
    frame=_frame({"transaction_id":"late","step":3},{"transaction_id":"first","step":1},{"transaction_id":"middle","step":2},{"transaction_id":"last","step":4},index=[10,"a","a",2])
    result=SplitTransactionRule().evaluate(frame)
    assert [e.transaction_id for e in result.evidence]==["late","first","middle","last"]
    assert len({e.transaction_id for e in result.evidence})==4
    assert "A" not in result.reason and all("B0" not in e.message for e in result.evidence)


def test_empty_missing_columns_immutability_repeatability():
    rule=SplitTransactionRule(); assert rule.evaluate(_frame().iloc[:0]).evidence==()
    with pytest.raises(ValueError): rule.evaluate(pd.DataFrame())
    frame=_frame({}, {}, {});before=frame.copy(deep=True)
    assert rule.evaluate(frame)==rule.evaluate(frame);assert_frame_equal(frame,before)


def test_non_dataframe_source_datetime_categorical_and_complete_evidence():
    rule=SplitTransactionRule(individual_amount_ceiling=10,cumulative_amount_threshold=15,min_count=2)
    with pytest.raises(TypeError,match="DataFrame"): rule.evaluate([])
    with pytest.raises(TypeError,match="source_row_id"): rule.evaluate(_frame({"source_row_id":True}))
    with pytest.raises(TypeError,match="datetime"): rule.evaluate(_frame({"transaction_datetime":"bad"}))
    categorical=_frame({},{});categorical["step"]=pd.Categorical([1,2])
    with pytest.raises(TypeError,match="categorical"): rule.evaluate(categorical)
    result=rule.evaluate(_frame({"transaction_id":" one ","source_row_id":pd.NA,"amount":8},{"transaction_id":"two","source_row_id":"row-2","amount":8}))
    assert [(e.transaction_id,e.source_row_id,e.actor_account,e.amount) for e in result.evidence]==[("one",None,"A",8.0),("two","row-2","A",8.0)]


@pytest.mark.parametrize("missing",REQUIRED)
def test_missing_required_columns(missing):
    with pytest.raises(ValueError,match=missing): SplitTransactionRule().evaluate(_frame().drop(columns=missing))


class _Fail(BaseRule):
    rule_id="fail-split";rule_name="Fail";description="Fails."
    def evaluate(self,transactions): raise RuntimeError("boom")


def test_registry_engine_failure_isolation():
    reg=RuleRegistry();reg.register(_Fail());reg.register(SplitTransactionRule())
    report=RuleEngine(reg).evaluate(_frame({}, {}, {}))
    assert [r.rule_id for r in report.results]==["split_transaction"]
    assert [e.rule_id for e in report.errors]==["fail-split"]
