"""
bankruptcy-fds/src/make_synth_case.py
가상 파산자 1인의 12개월 거래 데이터 생성기.

경일 님의 입력 3종 세트 설계를 따른다:
  1) transactions.csv  거래 CSV (+ 검증용 정답 라벨 is_suspicious, true_pattern)
  2) related_parties.csv  관계자 목록
  3) case_meta.csv  사건 메타 (파산 신청일 등)

정상 거래 위에 룰 패턴 4종(P01 반복송금, P02 분할송금,
P03 신청 직전 급증, P04 친족 고액)을 일부러 심는다.
라벨은 모델 학습에 쓰지 않고, 비지도 탐지 결과의 채점에만 쓴다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

FILING_DATE = pd.Timestamp("2026-06-30")   # 파산 신청일
START = FILING_DATE - pd.Timedelta(days=365)
DEBTOR = "DEBTOR01"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

MEMOS = ["월세", "식자재", "카드대금", "공과금", "거래처 결제", "급여", "보험료"]


def random_ts(n, day_lo=0, day_hi=365, hour_pool=range(8, 20)):
    days = rng.integers(day_lo, day_hi, n)
    hours = rng.choice(list(hour_pool), n)
    minutes = rng.integers(0, 60, n)
    return (START + pd.to_timedelta(days, "D")
            + pd.to_timedelta(hours, "h") + pd.to_timedelta(minutes, "m"))


def make_rows(ts, targets, amounts, memos, label, pattern):
    return pd.DataFrame({
        "timestamp": ts,
        "actor_account": DEBTOR,
        "target_account": targets,
        "amount": np.round(amounts, -2),
        "action_type": "transfer",
        "channel": rng.choice(["Mobile", "Bank", "ATM"], len(ts)),
        "memo": memos,
        "is_suspicious": label,
        "true_pattern": pattern,
    })


def main():
    frames = []

    # --- 정상 거래 660건: 평일 낮, 소액, 메모는 10% 공란 ---
    n = 660
    memos = rng.choice(MEMOS, n).astype(object)
    memos[rng.random(n) < 0.10] = np.nan          # 현실: 정상도 메모 공란 존재
    frames.append(make_rows(
        ts=random_ts(n),
        targets=[f"R{v:03d}" for v in rng.integers(0, 120, n)],
        amounts=np.exp(rng.normal(12.2, 0.8, n)),          # 수십만 원대
        memos=memos,
        label=0, pattern="NORMAL",
    ))

    # --- 교란 1: 심야·주말 생활 거래 40건 (정상) ---
    n = 40
    frames.append(make_rows(
        ts=random_ts(n, hour_pool=range(0, 24)),
        targets=[f"R{v:03d}" for v in rng.integers(0, 120, n)],
        amounts=np.exp(rng.normal(11.5, 0.7, n)),
        memos=[np.nan] * n,
        label=0, pattern="NORMAL",
    ))

    # --- 교란 2: 관계자 REL03(지인)에게 정당한 송금 3건 (정상) ---
    frames.append(make_rows(
        ts=random_ts(3, day_lo=60, day_hi=250),
        targets=["REL03"] * 3,
        amounts=rng.uniform(2e5, 6e5, 3),
        memos=["경조사비", "빌린 돈 상환", "경조사비"],
        label=0, pattern="NORMAL",
    ))

    # --- P01 반복송금: 관계자 REL01에게 25일간 8회 ---
    n = 8
    frames.append(make_rows(
        ts=random_ts(n, day_lo=280, day_hi=305),
        targets=["REL01"] * n,
        amounts=rng.uniform(1e6, 3e6, n),
        memos=[np.nan] * n,
        label=1, pattern="P01",
    ))

    # --- P02 분할송금: 같은 날 40분 내 4회, 총 600만원 ---
    base = START + pd.Timedelta(days=320, hours=14)
    frames.append(make_rows(
        ts=[base + pd.Timedelta(minutes=int(m)) for m in (0, 12, 25, 40)],
        targets=["R900"] * 4,
        amounts=np.full(4, 1.5e6),
        memos=[np.nan] * 4,
        label=1, pattern="P02",
    ))

    # --- P03 신청 직전 급증: 마지막 20일, 고액 10건, 심야 포함 ---
    n = 10
    frames.append(make_rows(
        ts=random_ts(n, day_lo=345, day_hi=365, hour_pool=range(0, 24)),
        targets=[f"R{v:03d}" for v in rng.integers(200, 210, n)],
        amounts=rng.uniform(3e6, 8e6, n),
        memos=[np.nan] * n,
        label=1, pattern="P03",
    ))

    # --- P04 친족 고액: 관계자 REL02에게 1,200만원 2회 ---
    frames.append(make_rows(
        ts=random_ts(2, day_lo=300, day_hi=340),
        targets=["REL02"] * 2,
        amounts=np.full(2, 1.2e7),
        memos=[np.nan] * 2,
        label=1, pattern="P04",
    ))

    tx = (pd.concat(frames, ignore_index=True)
            .sort_values("timestamp").reset_index(drop=True))

    # 잔액: 초기 2억 원에서 순차 차감 (행동로그의 balance_after)
    tx["balance_after"] = (2e8 - tx["amount"].cumsum()).round(-2)

    related = pd.DataFrame({
        "target_account": ["REL01", "REL02", "REL03"],
        "relation": ["친족", "친족", "지인"],
    })
    meta = pd.DataFrame({"case_id": ["CASE-2026-001"],
                         "debtor": [DEBTOR],
                         "filing_date": [FILING_DATE.date()]})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tx.to_csv(OUT_DIR / "transactions.csv", index=False)
    related.to_csv(OUT_DIR / "related_parties.csv", index=False)
    meta.to_csv(OUT_DIR / "case_meta.csv", index=False)

    print(f"생성 완료 → {OUT_DIR}")
    print(f"  거래 {len(tx):,}건 | 의심 {tx['is_suspicious'].sum()}건 "
          f"({tx['is_suspicious'].mean():.2%})")
    print(tx[tx.is_suspicious == 1]["true_pattern"].value_counts().to_string())


if __name__ == "__main__":
    main()
