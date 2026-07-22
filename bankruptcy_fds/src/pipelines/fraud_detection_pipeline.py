"""
bankruptcy-fds/src/pipelines/fraud_detection_pipeline.py
============================================================
FraudDetectionPipeline — 전체 흐름의 지휘자

[파이프라인이라는 설계의 의미]
1부 문서 7장의 데이터 흐름도가 그대로 코드가 된 것이다:

    거래 데이터 입력 (DataLoader: DB → DataFrame)
        ↓
    의심 규칙 적용 (rules 리스트를 순회)
        ↓
    위험 점수 계산 (RiskScorer)
        ↓
    판단 결과 저장 (DataLoader: DataFrame → DB)
        ↓
    보고서 반환 (관재인 검토)

[이 클래스에 판단 로직이 한 줄도 없는 이유]
파이프라인의 책임은 순서와 연결이지 판단이 아니다.
판단은 전부 규칙 객체 안에 있다. 그래서 이 파일은
새 규칙이 100개 추가되어도 단 한 글자도 바뀌지 않는다.
(rules 리스트에 객체 하나를 추가할 뿐이다)
이것이 BaseFraudRule 추상화가 지불한 비용의 보상이다.
"""

import json

import pandas as pd

from shared.scoring.risk_scorer import InvestigationReport, RiskScorer
from shared.utils.data_loader import DataLoader


class FraudDetectionPipeline:
    def __init__(self, loader: DataLoader, rules: list):
        """
        [의존성 주입(Dependency Injection)]
        DataLoader와 규칙 목록을 안에서 만들지 않고 밖에서 받는다.
        이유: 테스트 때는 가짜 loader와 규칙 하나만 주입하고,
        운영 때는 진짜 DB와 전체 규칙을 주입한다.
        부품을 갈아 끼울 수 있는 구조가 테스트 가능한 구조다.
        """
        self.loader = loader
        self.rules = rules
        self.scorer = RiskScorer()

    def run_for_debtor(self, debtor_row) -> InvestigationReport:
        """파산자 1명에 대한 전체 조사 절차."""
        debtor_id = debtor_row["debtor_id"]
        filing_date = pd.Timestamp(debtor_row["filing_date"])

        # 1) 장기 기억(DB) → 작업 기억(DataFrame)
        tx_df = self.loader.get_transactions(debtor_id)

        # 2) 모든 규칙을 순회 적용 — "많은 사건을 체계적으로 조사"하는 반복문
        results = [
            rule.evaluate(tx_df, filing_date=filing_date, debtor_id=debtor_id)
            for rule in self.rules
        ]

        # 3) 종합 판단
        report = self.scorer.aggregate(debtor_id, results)

        # 4) 판단을 다시 장기 기억에 남긴다 (감사 가능성)
        self._persist(debtor_id, results)

        return report

    def run_all(self) -> list[InvestigationReport]:
        """전체 파산자 일괄 조사."""
        debtors = self.loader.get_debtors()
        return [self.run_for_debtor(row) for _, row in debtors.iterrows()]

    def _persist(self, debtor_id: str, results: list) -> None:
        """RuleResult(객체 세계) → detection_results 테이블(DB 세계) 변환 저장."""
        rows = [{
            "debtor_id": debtor_id,
            "rule_name": r.rule_name,
            "risk_type": r.risk_type,
            "is_suspicious": int(r.is_suspicious),
            "risk_score": r.risk_score,
            "reason": r.reason,
            "evidence_ids": json.dumps(r.evidence_ids, ensure_ascii=False),
        } for r in results]
        self.loader.save_results(pd.DataFrame(rows))
