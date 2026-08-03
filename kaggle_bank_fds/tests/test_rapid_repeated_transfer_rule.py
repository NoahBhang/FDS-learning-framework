"""Contract tests for RapidRepeatedTransferRule."""

from decimal import Decimal
import inspect
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kaggle_bank_fds.src.rules.base_rule import BaseRule
from kaggle_bank_fds.src.rules.rapid_repeated_transfer_rule import RapidRepeatedTransferRule
from kaggle_bank_fds.src.rules.rule_engine import RuleEngine
from kaggle_bank_fds.src.rules.rule_registry import RuleRegistry


REQUIRED=RapidRepeatedTransferRule.REQUIRED_COLUMNS
def _frame(*changes,index=None):
    rows=[]
    for i,c in enumerate(changes or ({},)):
        rows.append({"transaction_id":f"tx-{i}","source_row_id":i,"step":i,
                     "transaction_datetime":pd.Timestamp("2026-01-01"),"action_type":"TRANSFER",
                     "amount":40_000.0,"actor_account":"A","target_account":"B"}|c)
    return pd.DataFrame(rows,columns=REQUIRED,index=index)


def test_metadata_signature_defaults_readonly():
    rule=RapidRepeatedTransferRule(); sig=inspect.signature(RapidRepeatedTransferRule)
    assert rule.rule_id=="rapid_repeated_transfer"
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values())
    assert (rule.max_step_gap,rule.min_count,rule.min_total_amount,rule.score)==(24,3,100_000.0,30)
    with pytest.raises(AttributeError): rule.min_count=2


def test_numpy_parameters_normalized():
    rule=RapidRepeatedTransferRule(max_step_gap=np.int64(2),min_count=np.int64(2),min_total_amount=np.float64(1),score=np.int64(9))
    assert all(type(v) is int for v in (rule.max_step_gap,rule.min_count,rule.score)); assert type(rule.min_total_amount) is float


@pytest.mark.parametrize(("name","value"),[("max_step_gap",True),("max_step_gap","1"),("min_count",np.bool_(True)),("min_count",2.0),("min_total_amount",True),("min_total_amount","1"),("min_total_amount",Decimal("1")),("min_total_amount",1+0j),("score",True),("score","1"),("score",1.0),("score",None)])
def test_invalid_parameter_types(name,value):
    with pytest.raises(TypeError): RapidRepeatedTransferRule(**{name:value})


@pytest.mark.parametrize("kwargs",[{"max_step_gap":-1},{"min_count":1},{"min_total_amount":0},{"min_total_amount":np.nan},{"min_total_amount":np.inf},{"score":0},{"score":101}])
def test_invalid_parameter_values(kwargs):
    with pytest.raises(ValueError): RapidRepeatedTransferRule(**kwargs)


def test_min_count_and_total_boundaries():
    assert not RapidRepeatedTransferRule().evaluate(_frame({},{})).triggered
    assert RapidRepeatedTransferRule().evaluate(_frame({}, {}, {})).triggered
    below=_frame({"amount":30_000},{"amount":30_000},{"amount":39_999})
    exact=_frame({"amount":30_000},{"amount":30_000},{"amount":40_000})
    assert not RapidRepeatedTransferRule().evaluate(below).triggered
    assert RapidRepeatedTransferRule().evaluate(exact).triggered


def test_gap_boundary_tie_and_overflow():
    rule=RapidRepeatedTransferRule(min_total_amount=3)
    assert rule.evaluate(_frame({"step":0,"amount":1},{"step":24,"amount":1},{"step":24,"amount":1})).triggered
    assert not rule.evaluate(_frame({"step":0,"amount":1},{"step":25,"amount":1},{"step":25,"amount":1})).triggered
    assert rule.evaluate(_frame({"step":1,"amount":1},{"step":1,"amount":1},{"step":1,"amount":1})).triggered


def test_pair_group_boundaries():
    rule=RapidRepeatedTransferRule(min_total_amount=3)
    assert not rule.evaluate(_frame({"amount":1,"target_account":"B1"},{"amount":1,"target_account":"B2"},{"amount":1,"target_account":"B3"})).triggered
    assert not rule.evaluate(_frame({"amount":1,"actor_account":"A1"},{"amount":1,"actor_account":"A2"},{"amount":1,"actor_account":"A3"})).triggered


def test_nonpositive_and_non_candidate_are_not_detected():
    rule=RapidRepeatedTransferRule(min_total_amount=1,min_count=2)
    frame=_frame({"amount":0},{"amount":-1},{"action_type":"PAYMENT","amount":"bad","step":"bad","transaction_id":"","actor_account":None,"target_account":None})
    assert not rule.evaluate(frame).triggered


@pytest.mark.parametrize(("field","value","error"),[("amount",pd.NA,ValueError),("amount",np.inf,ValueError),("amount",True,TypeError),("amount","1",TypeError),("step",pd.NA,ValueError),("step",True,TypeError),("step",1.0,TypeError),("actor_account",None,ValueError),("target_account"," ",ValueError),("transaction_id","",ValueError)])
def test_invalid_candidate_fields(field,value,error):
    with pytest.raises(error): RapidRepeatedTransferRule().evaluate(_frame({field:value}))


def test_duplicate_transaction_rejected_source_allowed():
    with pytest.raises(ValueError,match="unique"): RapidRepeatedTransferRule().evaluate(_frame({"transaction_id":"x"},{"transaction_id":"x"}))
    result=RapidRepeatedTransferRule(min_count=2,min_total_amount=2).evaluate(_frame({"source_row_id":7,"amount":1},{"source_row_id":7,"amount":1}))
    assert [e.source_row_id for e in result.evidence]==[7,7]


def test_overlapping_windows_union_and_original_order():
    frame=_frame({"transaction_id":"late","step":3,"amount":40_000},{"transaction_id":"first","step":1,"amount":40_000},{"transaction_id":"middle","step":2,"amount":40_000},{"transaction_id":"last","step":4,"amount":40_000},index=[9,2,2,"x"])
    result=RapidRepeatedTransferRule().evaluate(frame)
    assert [e.transaction_id for e in result.evidence]==["late","first","middle","last"]
    assert len({e.transaction_id for e in result.evidence})==4


def test_empty_missing_columns_and_immutability_repeatability():
    rule=RapidRepeatedTransferRule(); empty=_frame().iloc[:0]
    assert rule.evaluate(empty).evidence==()
    with pytest.raises(ValueError): rule.evaluate(pd.DataFrame())
    frame=_frame({}, {}, {}); before=frame.copy(deep=True)
    assert rule.evaluate(frame)==rule.evaluate(frame); assert_frame_equal(frame,before)


def test_non_dataframe_source_datetime_categorical_and_complete_evidence():
    rule=RapidRepeatedTransferRule(min_count=2,min_total_amount=2)
    with pytest.raises(TypeError,match="DataFrame"): rule.evaluate([])
    with pytest.raises(TypeError,match="source_row_id"): rule.evaluate(_frame({"source_row_id":True}))
    with pytest.raises(TypeError,match="datetime"): rule.evaluate(_frame({"transaction_datetime":"bad"}))
    categorical=_frame({},{});categorical["amount"]=pd.Categorical([1,1])
    with pytest.raises(TypeError,match="categorical"): rule.evaluate(categorical)
    result=rule.evaluate(_frame({"transaction_id":" one ","source_row_id":pd.NA,"amount":1},{"transaction_id":"two","source_row_id":"row-2","amount":1}))
    assert [(e.transaction_id,e.source_row_id,e.actor_account,e.target_account,e.amount) for e in result.evidence]==[("one",None,"A","B",1.0),("two","row-2","A","B",1.0)]


@pytest.mark.parametrize("missing",REQUIRED)
def test_missing_required_columns(missing):
    with pytest.raises(ValueError,match=missing): RapidRepeatedTransferRule().evaluate(_frame().drop(columns=missing))


class _Fail(BaseRule):
    rule_id="fail-rapid";rule_name="Fail";description="Fails."
    def evaluate(self,transactions): raise RuntimeError("boom")


def test_registry_engine_failure_isolation():
    reg=RuleRegistry();reg.register(_Fail());reg.register(RapidRepeatedTransferRule())
    report=RuleEngine(reg).evaluate(_frame({}, {}, {}))
    assert [r.rule_id for r in report.results]==["rapid_repeated_transfer"]
    assert [e.rule_id for e in report.errors]==["fail-rapid"]
