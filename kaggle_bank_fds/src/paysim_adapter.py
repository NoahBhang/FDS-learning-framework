"""
kaggle-bank-fds/src/paysim_adapter.py
PaySim CSV를 행동로그 표준 스키마로 번역하는 어댑터.

파산관재인 FDS와 동일한 피처 파이프라인(shared/features.py)을
재사용하기 위한 계층이다. 데이터셋이 바뀌어도 이 파일만 갈아끼우면
나머지 파이프라인은 그대로 동작한다.
"""

import pandas as pd

# PaySim의 step은 "시뮬레이션 시작 후 경과 시간(시간 단위)"이므로
# 임의의 기준일에 더해 timestamp로 변환한다.
BASE_DATE = pd.Timestamp("2026-01-01")

# PaySim 컬럼 → 행동로그 표준 스키마
RENAME = {
    "nameOrig": "actor_account",
    "nameDest": "target_account",
    "type": "action_type",
    "newbalanceOrig": "balance_after",
}

# isFlaggedFraud는 PaySim 내부 규칙의 판정 결과(사실상 정답의 일부)이므로
# 피처로 쓰면 데이터 누수가 된다. 반드시 제거한다.
LEAK_COLUMNS = ["isFlaggedFraud"]


def load_paysim(path: str,
                fraud_types_only: bool = True,
                max_step: int | None = None) -> pd.DataFrame:
    """PaySim CSV 로드.

    fraud_types_only: PaySim의 사기는 TRANSFER와 CASH_OUT에서만
        발생하므로, 두 유형만 남겨 문제를 명확히 한다 (약 277만 행).
    max_step: 첫 실행 검증용. 예: 300이면 처음 300시간 분량만 사용
        (시간 순서를 보존하는 축소라서 무작위 샘플링보다 안전하다).
    """
    df = pd.read_csv(path)
    if fraud_types_only:
        df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    if max_step is not None:
        df = df[df["step"] <= max_step]
    return df.reset_index(drop=True)


def to_action_log(df: pd.DataFrame) -> pd.DataFrame:
    """PaySim → 행동로그 스키마 + PaySim 고유 신호 피처."""
    out = df.rename(columns=RENAME).copy()
    out["timestamp"] = BASE_DATE + pd.to_timedelta(out["step"], unit="h")

    # --- PaySim 고유 신호 1: 잔액 부등식 오차 ---
    # 정상 거래라면 (이전 잔액 - 금액 = 이후 잔액)이 성립해야 한다.
    # 이 등식이 깨진 정도 자체가 강력한 사기 신호로 알려져 있다.
    out["balance_error_orig"] = (
        out["oldbalanceOrg"] - out["amount"] - out["balance_after"]
    )
    out["balance_error_dest"] = (
        out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    )

    # --- PaySim 고유 신호 2: 잔액 소진 ---
    # 파산 FDS의 "잔액소진 + 신청 직전" 조합 룰과 같은 개념.
    out["orig_emptied"] = (
        (out["oldbalanceOrg"] > 0) & (out["balance_after"] == 0)
    ).astype(int)

    out["is_transfer"] = (out["action_type"] == "TRANSFER").astype(int)

    return out.drop(columns=[c for c in LEAK_COLUMNS if c in out.columns])
