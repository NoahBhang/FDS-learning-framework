"""
bankruptcy_fds/src/models.py
============================================================
predict() — DB 없이 거래 DataFrame 하나로 즉시 판단을 돌려주는 진입점

[왜 필요한가]
FraudDetectionPipeline(pipelines/fraud_detection_pipeline.py)은
DataLoader(DB)를 전제로 "파산자 전체 일괄 조사"를 담당한다.
반면 서빙/실험 환경에서는 이미 메모리에 있는 거래 DataFrame 하나에 대해
바로 점수를 매겨야 하는 경우가 많다(API 엔드포인트, 노트북 실험 등).

이 파일은 그 얇은 진입점이다. 판단 로직은 새로 만들지 않는다 —
rules/ 아래 다섯 규칙(BaseFraudRule 하위 클래스)을 그대로 재사용하고,
RiskScorer로 종합한다. "판단 로직은 한 곳에만 존재한다"는 원칙을
지키기 위해 파이프라인과 동일한 규칙·스코어러를 그대로 불러 쓴다.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from shared.scoring.risk_scorer import RiskScorer               # noqa: E402
from bankruptcy_fds.src.rules.split_transfer import SplitTransferRule    # noqa: E402
from bankruptcy_fds.src.rules.related_party import RelatedPartyRule      # noqa: E402
from bankruptcy_fds.src.rules.layering import LayeringRule               # noqa: E402
from bankruptcy_fds.src.rules.velocity_spike import VelocitySpikeRule    # noqa: E402
from bankruptcy_fds.src.rules.temporal_anomaly import TemporalAnomalyRule  # noqa: E402

logger = logging.getLogger(__name__)

# 기본 규칙 세트 — rules/ 아래 5종 전부. FraudDetectionPipeline은 규칙 목록을
# 외부(예: scripts/run_demo.py)에서 주입받으므로 이 목록과 반드시 일치하지는
# 않는다 — predict()만의 기본값이다.
DEFAULT_RULES = [
    SplitTransferRule(),
    RelatedPartyRule(),
    LayeringRule(),
    VelocitySpikeRule(),
    TemporalAnomalyRule(),
]


def predict(transaction_data: pd.DataFrame,
            filing_date: pd.Timestamp = None,
            rules: list = None) -> dict:
    """
    한 파산자의 거래 DataFrame을 규칙 세트로 평가하여 종합 판단을 반환한다.

    Parameters
    ----------
    transaction_data : transactions 테이블과 동일한 컬럼을 가진 DataFrame
                        (transaction_id, debtor_id, transaction_date, amount,
                         sender, receiver, relation_type, transaction_type).
                        단일 파산자(debtor_id 하나)의 거래만 담아야 한다 —
                        LayeringRule의 BFS가 그 debtor_id를 출발 노드로 삼는다.
    filing_date : 조사 기준 시점(파산 신청일에 해당). 생략 시 데이터 내
                  최신 거래일을 기준점으로 삼는다 — "지금까지의 거래를
                  실시간으로 심사한다"는 서빙 시나리오의 자연스러운 기본값.
    rules : 평가에 쓸 규칙 목록. 생략 시 DEFAULT_RULES(쪼개기 송금·편파 변제·
            레이어링·거래 급증·비정상 시간대) 전부를 쓴다. 파이프라인과 같은
            DI 방식 — 테스트에서 규칙 하나만 주입해 검증하고 싶을 때를 위한
            확장점이다.

    Returns
    -------
    dict
        fraud_score      : 0.0~1.0 종합 위험 점수 (RiskScorer의 0~100점을 정규화).
                            평가에 실패한 규칙은 집계에서 제외된다.
        triggered_rules  : 의심 판정된 규칙 이름(rule_name) 목록.
                            평가에 실패한 규칙은 포함되지 않는다.
        details          : 규칙별 상세 판단 +
                            {rule_name: {risk_type, is_suspicious, risk_score,
                                         reason, evidence_ids}, ...,
                             "skipped_rules": {rule_name: 에러 메시지, ...}}

    알려진 제약
    -----------
    - 단일 파산자 입력만 지원한다. transaction_data에 debtor_id가
      둘 이상 섞이면 ValueError를 던진다(LayeringRule의 BFS가
      단일 출발 노드를 전제하기 때문).
    - filing_date를 생략하면 데이터 내 최신 거래일을 대리 기준점으로
      쓴다. 실제 파산 신청일과 다를 수 있으므로, 정확한 심사가
      필요하면 반드시 filing_date를 명시적으로 넘겨야 한다.
    - 규칙 하나가 예외를 던지면 그 규칙은 조용히 건너뛰고 나머지
      규칙으로 계속 진행한다(가용성 우선). 그 결과 fraud_score가
      실제보다 낮게 나올 수 있다 — 반환된 details["skipped_rules"]가
      비어 있지 않다면 그 판단은 불완전하다는 뜻이므로 반드시 확인해야
      한다.
    - transaction_data에 필요한 컬럼(스키마 참고: transaction_id,
      debtor_id, transaction_date, amount, sender, receiver,
      relation_type, transaction_type)이 없으면 predict() 자체가
      아니라 해당 컬럼을 쓰는 개별 규칙만 실패하며 skipped_rules로
      드러난다 — 사전 스키마 검증은 하지 않는다.
    """
    if transaction_data.empty:
        raise ValueError("transaction_data가 비어 있다.")

    debtor_ids = transaction_data["debtor_id"].unique()
    if len(debtor_ids) != 1:
        raise ValueError(
            f"predict()는 단일 파산자의 거래만 받는다. "
            f"debtor_id {len(debtor_ids)}종이 섞여 있다: {list(debtor_ids)}"
        )
    debtor_id = debtor_ids[0]

    if filing_date is None:
        filing_date = pd.Timestamp(transaction_data["transaction_date"].max())
    else:
        filing_date = pd.Timestamp(filing_date)

    active_rules = rules if rules is not None else DEFAULT_RULES

    # 규칙 하나의 실패가 전체 판단을 막지 않도록 개별적으로 감싼다.
    # (파이프라인 분석에서 지적된 "예외 처리 없음" 위험에 대한 방어)
    results = []
    skipped_rules = {}
    for rule in active_rules:
        try:
            results.append(
                rule.evaluate(transaction_data, filing_date=filing_date, debtor_id=debtor_id)
            )
        except Exception as exc:
            logger.error("규칙 '%s' 평가 실패, 건너뛴다: %s", rule.rule_name, exc)
            skipped_rules[rule.rule_name] = str(exc)

    report = RiskScorer().aggregate(debtor_id, results)

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
