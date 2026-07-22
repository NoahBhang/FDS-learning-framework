"""
scripts/run_demo.py
============================================================
전체 시스템 실행 데모 — 처음부터 끝까지 한 번에

실행 순서 (곧 시스템의 생애 주기):
  1. 샘플 데이터 생성 (현실 세계를 CSV로 흉내)
  2. DB 스키마 생성 (장기 기억 준비)
  3. CSV → DB 적재
  4. 파이프라인 실행 (DB → DataFrame → 규칙 → 판단)
  5. 판단 결과 DB 저장 + 관재인 보고서 출력
  6. 은행권 FDS 규칙도 같은 프레임워크로 실행

실행 방법 (프로젝트 루트에서):
  python scripts/run_demo.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가.
# 이유: shared가 pip 패키지가 아니라 로컬 폴더이므로
# 파이썬에게 "여기서부터 찾아라"를 알려 주어야 한다.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from shared.utils.data_loader import DataLoader


# ------------------------------------------------------------
# [현실 수업: 패키지 이름 충돌]
# bankruptcy_fds와 kaggle_bank_fds 둘 다 'src'라는 패키지를 갖는다.
# 파이썬은 한 이름의 모듈을 한 번만 기억하므로(sys.modules 캐시),
# 두 'src'를 동시에 일반 import하면 하나가 다른 하나를 가린다.
# 해법: importlib로 "이 파일을 이 이름으로 읽어라"라고 명시한다.
# (실무 대안: 각 프로젝트를 자기 폴더 안에서 실행하거나,
#  패키지 이름을 bankruptcy_src처럼 고유하게 짓는다)
# ------------------------------------------------------------
import importlib.util


def load_module(alias: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(alias, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


BK = ROOT / "bankruptcy_fds" / "src"
split_mod = load_module("bk_split", BK / "rules" / "split_transfer.py")
related_mod = load_module("bk_related", BK / "rules" / "related_party.py")
layering_mod = load_module("bk_layering", BK / "rules" / "layering.py")
pipeline_mod = load_module("bk_pipeline", BK / "pipelines" / "fraud_detection_pipeline.py")

SplitTransferRule = split_mod.SplitTransferRule
RelatedPartyRule = related_mod.RelatedPartyRule
LayeringRule = layering_mod.LayeringRule
FraudDetectionPipeline = pipeline_mod.FraudDetectionPipeline


# ============================================================
# 1. 샘플 데이터 생성 — 시나리오가 담긴 가짜 현실
# ============================================================
def generate_sample_data(data_dir: Path):
    """
    두 명의 파산자를 만든다.
      D001 홍길동: 세 가지 수법을 모두 쓴 고위험 시나리오
      D002 김철수: 평범한 생활비 지출만 있는 정상 시나리오
    FDS가 D001만 정확히 적발하고 D002를 건드리지 않아야
    시스템이 올바르다 (오탐 없는 정탐).
    """
    debtors = pd.DataFrame([
        {"debtor_id": "D001", "name": "홍길동", "filing_date": "2026-06-01"},
        {"debtor_id": "D002", "name": "김철수", "filing_date": "2026-06-01"},
    ])

    tx = []
    # --- D001 시나리오 1: 쪼개기 송금 (신청 전 30일 내, 김OO에게 7회) ---
    for i, (day, amt) in enumerate([
        (5, 900_000), (8, 950_000), (12, 800_000), (15, 990_000),
        (18, 850_000), (22, 900_000), (25, 910_000),
    ]):
        tx.append({
            "transaction_id": f"T{i+1:04d}", "debtor_id": "D001",
            "transaction_date": f"2026-05-{day:02d}", "amount": amt,
            "sender": "D001", "receiver": "김OO",
            "relation_type": "가족", "transaction_type": "송금",
        })

    # --- D001 시나리오 2: 편파 변제 (지인 이OO에게 대액) ---
    tx.append({
        "transaction_id": "T0008", "debtor_id": "D001",
        "transaction_date": "2026-04-10", "amount": 8_000_000,
        "sender": "D001", "receiver": "이OO",
        "relation_type": "지인", "transaction_type": "송금",
    })

    # --- D001 시나리오 3: 레이어링 (김OO → 김OO배우자 → 박OO법인) ---
    tx.append({
        "transaction_id": "T0009", "debtor_id": "D001",
        "transaction_date": "2026-05-26", "amount": 5_000_000,
        "sender": "김OO", "receiver": "김OO배우자",
        "relation_type": "가족", "transaction_type": "송금",
    })
    tx.append({
        "transaction_id": "T0010", "debtor_id": "D001",
        "transaction_date": "2026-05-28", "amount": 4_800_000,
        "sender": "김OO배우자", "receiver": "박OO법인",
        "relation_type": "법인", "transaction_type": "송금",
    })

    # --- D002 시나리오: 정상 거래 (마트, 통신비 등) ---
    for i, (day, receiver, amt) in enumerate([
        (3, "OO마트", 150_000), (10, "OO통신", 89_000),
        (17, "OO카드", 450_000), (24, "OO보험", 120_000),
    ]):
        tx.append({
            "transaction_id": f"T1{i+1:03d}", "debtor_id": "D002",
            "transaction_date": f"2026-05-{day:02d}", "amount": amt,
            "sender": "D002", "receiver": receiver,
            "relation_type": "법인", "transaction_type": "카드결제",
        })

    data_dir.mkdir(parents=True, exist_ok=True)
    debtors.to_csv(data_dir / "sample_debtors.csv", index=False)
    pd.DataFrame(tx).to_csv(data_dir / "sample_transactions.csv", index=False)
    print(f"✔ 샘플 데이터 생성: 파산자 {len(debtors)}명, 거래 {len(tx)}건")


# ============================================================
# 2~5. 파산관재인 FDS 실행
# ============================================================
def run_bankruptcy_fds():
    print("\n" + "=" * 60)
    print("  파산관재인 FDS 실행")
    print("=" * 60)

    data_dir = ROOT / "bankruptcy_fds" / "data"
    db_path = data_dir / "fds.db"
    if db_path.exists():
        db_path.unlink()   # 데모의 재실행 가능성을 위해 초기화

    generate_sample_data(data_dir)

    # 장기 기억(DB) 준비: 스키마 생성 → CSV 적재
    loader = DataLoader(f"sqlite:///{db_path}")
    loader.init_schema(str(ROOT / "database" / "schema.sql"))
    loader.load_csv_to_table(str(data_dir / "sample_debtors.csv"), "debtors")
    loader.load_csv_to_table(str(data_dir / "sample_transactions.csv"), "transactions")
    print("✔ DB 적재 완료 (SQLite)")

    # 규칙 조립: 새 규칙이 생기면 이 리스트에 한 줄 추가하면 끝이다.
    rules = [
        SplitTransferRule(window_days=30, min_count=5,
                          total_amount_threshold=5_000_000),
        RelatedPartyRule(window_days=90, ratio_threshold=0.5),
        LayeringRule(min_hops=2, min_amount=500_000),
    ]

    pipeline = FraudDetectionPipeline(loader, rules)
    reports = pipeline.run_all()

    # 관재인 보고서 출력 — "사용자 인터페이스 세계"의 최소 형태
    for rp in reports:
        print("\n" + "-" * 60)
        print(f"[파산자 {rp.debtor_id}]  종합 위험 점수: {rp.total_score}점 ({rp.grade})")
        print(f"  요약: {rp.summary}")
        for f in rp.findings:
            print(f"\n  ▶ {f.risk_type} (+{f.risk_score}점)")
            print(f"    근거: {f.reason}")
            print(f"    증거 거래: {f.evidence_ids}")


# ============================================================
# 6. 은행권 FDS 실행 — 같은 프레임워크의 재사용 증명
# ============================================================
def run_bank_fds():
    print("\n" + "=" * 60)
    print("  은행권 FDS 실행 (PaySim 형식 샘플)")
    print("=" * 60)

    bank_mod = load_module(
        "bank_rules",
        ROOT / "kaggle_bank_fds" / "src" / "rules" / "bank_fraud_rules.py",
    )
    TransferCashOutRule = bank_mod.TransferCashOutRule
    FullBalanceTransferRule = bank_mod.FullBalanceTransferRule

    # PaySim 형식의 미니 샘플 (실전에서는 Kaggle CSV를 그대로 읽는다)
    df = pd.DataFrame([
        # 정상 결제
        {"step": 1, "type": "PAYMENT",  "amount": 9_800,
         "nameOrig": "C1001", "oldbalanceOrg": 170_000, "newbalanceOrig": 160_200,
         "nameDest": "M2001"},
        # 의심 패턴: C1002가 잔액 전액을 C9999로 이체 →
        #            C9999가 2시간 뒤 거의 같은 금액을 현금 인출
        {"step": 5, "type": "TRANSFER", "amount": 1_500_000,
         "nameOrig": "C1002", "oldbalanceOrg": 1_500_000, "newbalanceOrig": 0,
         "nameDest": "C9999"},
        {"step": 7, "type": "CASH_OUT", "amount": 1_480_000,
         "nameOrig": "C9999", "oldbalanceOrg": 1_500_000, "newbalanceOrig": 20_000,
         "nameDest": "C0000"},
    ])

    # 파산 FDS와 완전히 같은 사용법: 규칙 목록 순회 → RuleResult
    rules = [TransferCashOutRule(), FullBalanceTransferRule()]
    for rule in rules:
        result = rule.evaluate(df)
        mark = "⚠ 의심" if result.is_suspicious else "정상"
        print(f"\n  [{rule.risk_type}] → {mark} (+{result.risk_score}점)")
        print(f"    근거: {result.reason}")


if __name__ == "__main__":
    run_bankruptcy_fds()
    run_bank_fds()
    print("\n" + "=" * 60)
    print("  데모 완료. detection_results 테이블에서 판단 기록을 확인할 수 있다.")
    print("=" * 60)
