"""
kaggle_bank_fds/src/models/predictor.py
============================================================
predict() — DB/학습 파이프라인 없이 거래 DataFrame 하나로 즉시 판단을 돌려주는 진입점

[bankruptcy_fds/src/models.py와의 관계]
파산관재인 FDS 쪽 predict()와 반환 형태·예외 처리 방식이 동일하다 —
"같은 사고틀, 다른 도메인"이라는 이 저장소의 핵심 메시지를 서빙
계층에서도 반복하는 것이다. 차이는 도메인 규칙뿐이다:
파산 FDS는 파산자 1명 단위(LayeringRule의 BFS가 출발 노드를 요구)로
스코어링하지만, 은행권 FDS(PaySim)의 두 규칙은 self-merge/전체 집계
기반이라 특정 계좌 하나로 입력을 제한할 필요가 없다 — 배치 전체를
한 번에 평가한다.

판단 로직은 새로 만들지 않는다. rules/bank_fraud_rules.py의 두 규칙
(BaseFraudRule 하위 클래스)과 shared.scoring.RiskScorer를 그대로
재사용한다.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[3]))

from shared.scoring.risk_scorer import RiskScorer                    # noqa: E402
from kaggle_bank_fds.src.rules.bank_fraud_rules import (             # noqa: E402
    TransferCashOutRule, FullBalanceTransferRule,
)

logger = logging.getLogger(__name__)

# 기본 규칙 세트 — 둘 다 배치 전체(여러 계좌)를 대상으로 self-merge/집계한다.
DEFAULT_RULES = [TransferCashOutRule(), FullBalanceTransferRule()]


def predict(transaction_data: pd.DataFrame, rules: list = None) -> dict:
    """
    은행 거래 배치(PaySim 스키마)를 규칙 세트로 평가해 종합 판단을 반환한다.

    Parameters
    ----------
    transaction_data : PaySim 원본 컬럼(type, amount, nameOrig, nameDest,
                        oldbalanceOrg, step, ...)을 가진 DataFrame.
                        bankruptcy_fds.src.models.predict()와 달리 단일
                        엔티티로 제한하지 않는다 — TransferCashOutRule이
                        계좌 간 이체→인출 연쇄를 self-merge로 찾아내려면
                        여러 계좌가 섞인 배치 전체가 필요하기 때문이다.
                        원본 인덱스를 보존해야 한다: PaySim에는 거래 고유
                        ID 컬럼이 없어 두 규칙 모두 evidence_ids로
                        DataFrame의 행 인덱스를 그대로 쓴다.
    rules : 평가에 쓸 규칙 목록. 생략 시 DEFAULT_RULES(이체 후 즉시
            현금화, 계좌 전액 이체) 전부를 쓴다. bankruptcy_fds 쪽과
            같은 DI 방식 — 테스트에서 규칙 하나만 주입해 검증할 때
            쓰는 확장점이다.

    Returns
    -------
    dict
        fraud_score      : 0.0~1.0 종합 위험 점수 (RiskScorer의 0~100점을
                            정규화). 평가에 실패한 규칙은 집계에서 제외된다.
        triggered_rules  : 의심 판정된 규칙 이름(rule_name) 목록.
                            평가에 실패한 규칙은 포함되지 않는다.
        details          : 규칙별 상세 판단 +
                            {rule_name: {risk_type, is_suspicious, risk_score,
                                         reason, evidence_ids}, ...,
                             "skipped_rules": {rule_name: 에러 메시지, ...}}

    알려진 제약
    -----------
    - bankruptcy_fds.src.models.predict()와 달리 단일 엔티티로 입력을
      제한하지 않는다. 배치에 여러 계좌가 섞여도 정상 동작하지만,
      그만큼 이 함수는 "배치 전체에 대한 종합 판단" 하나만 반환한다 —
      계좌별로 보려면 호출 전에 직접 계좌 단위로 나눠 여러 번 호출해야
      한다.
    - PaySim 데이터에는 거래 고유 ID가 없어 evidence_ids는 DataFrame
      행 인덱스다. 인덱스를 초기화(reset_index)한 복사본을 넘기면
      근거 거래가 원본과 어긋난다.
    - 규칙 하나가 예외를 던지면 그 규칙은 조용히 건너뛰고 나머지
      규칙으로 계속 진행한다(가용성 우선). 그 결과 fraud_score가
      실제보다 낮게 나올 수 있다 — 반환된 details["skipped_rules"]가
      비어 있지 않다면 그 판단은 불완전하다는 뜻이므로 반드시 확인해야
      한다.
    - 라벨(isFraud/isFlaggedFraud)은 채점 전용이며 이 함수는 참조하지
      않는다 — 규칙이 실서비스에서 라벨 없이 동작해야 한다는 전제를
      지킨다.
    """
    if transaction_data.empty:
        raise ValueError("transaction_data가 비어 있다.")

    active_rules = rules if rules is not None else DEFAULT_RULES

    # 규칙 하나의 실패가 전체 판단을 막지 않도록 개별적으로 감싼다.
    # (bankruptcy_fds.src.models.predict()와 동일한 방어 방식)
    results = []
    skipped_rules = {}
    for rule in active_rules:
        try:
            results.append(rule.evaluate(transaction_data))
        except Exception as exc:
            logger.error("규칙 '%s' 평가 실패, 건너뛴다: %s", rule.rule_name, exc)
            skipped_rules[rule.rule_name] = str(exc)

    report = RiskScorer().aggregate("batch", results)

    details = {
        r.rule_name: {
            "risk_type": r.risk_type,
            "is_suspicious": r.is_suspicious,
            "risk_score": r.risk_score,
            "reason": r.reason,
            "evidence_ids": r.evidence_ids,
        }
        for r in results
    }
    details["skipped_rules"] = skipped_rules

    return {
        "fraud_score": report.total_score / 100.0,
        "triggered_rules": [r.rule_name for r in report.findings],
        "details": details,
    }
