"""
scripts/app.py
============================================================
predict() 스모크 테스트용 Streamlit 앱

DB(schema.sql) 없이, 거래 CSV 하나만 업로드하면 즉시 predict()로
위험 점수를 매겨 보여준다. 사이드바 라디오로 두 프로젝트 중 하나를
고를 수 있다 — "같은 사고틀, 다른 도메인"이라는 이 저장소의 핵심
메시지를 한 화면에서 나란히 확인하기 위함이다.

  파산관재인 FDS : bankruptcy_fds.src.models.predict()
                   (파산자 1명 단위, filing_date 기준 시간창 룰)
  은행권 FDS     : kaggle_bank_fds.src.models.predictor.predict()
                   (계좌 여러 개가 섞인 배치 전체를 한 번에 평가)

bankruptcy_fds/src/ui/streamlit_app.py(DB 기반 대시보드)와 달리
DataLoader 없이 predict()의 서빙 경로만 확인하는 용도다.

실행: streamlit run scripts/app.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="FDS Model Test", layout="wide")
st.title("FDS Model Test")
st.write("`predict()` 스모크 테스트 — 거래 CSV를 업로드하면 위험 점수를 매깁니다.")

mode = st.sidebar.radio("FDS 선택", ["파산관재인 FDS", "은행권 FDS"])
st.sidebar.divider()


def render_details(details: dict, source_tx: pd.DataFrame, evidence_col: str | None):
    """규칙별 상세 판단 + skipped_rules를 두 모드가 공유하는 렌더러.

    details에는 "skipped_rules"라는, 다른 항목과 모양이 다른 키가
    섞여 있으므로(그 값은 {규칙명: 에러 메시지}) 먼저 분리해야 한다 —
    그냥 순회하면 이 항목도 규칙 판단인 것처럼 잘못 렌더링된다.
    """
    skipped = details.get("skipped_rules", {})
    rule_details = {k: v for k, v in details.items() if k != "skipped_rules"}

    for rule_name, detail in rule_details.items():
        icon = "⚠" if detail["is_suspicious"] else "✅"
        with st.expander(f"{icon} {detail['risk_type']} (+{detail['risk_score']}점)"):
            st.write(detail["reason"])
            if detail["evidence_ids"]:
                if evidence_col:
                    ev = source_tx[source_tx[evidence_col].isin(detail["evidence_ids"])]
                else:
                    # PaySim처럼 거래 고유 ID가 없는 경우: evidence_ids는
                    # 원본 DataFrame의 행 인덱스 자체다.
                    ev = source_tx.loc[source_tx.index.isin(detail["evidence_ids"])]
                st.dataframe(ev, use_container_width=True)

    if skipped:
        st.warning(f"평가에 실패해 건너뛴 규칙 {len(skipped)}건 — 판단이 불완전할 수 있습니다.")
        st.json(skipped)


def render_downloads(source_tx: pd.DataFrame, result: dict, model_slug: str,
                      id_col: str, entity_col: str | None):
    """CSV(거래 단위 결과) + JSON(규칙별 상세 판단) 다운로드 버튼.

    fraud_score·triggered_rules는 배치/개체 전체에 대한 종합값이므로
    CSV의 모든 행에 동일하게 복제된다 — "거래 하나하나의 점수"가
    아니라 "이 조사 대상 전체의 점수를 거래 단위로 펼친 표"라는 뜻이다.
    개별 거래가 어떤 규칙의 근거였는지는 위 상세 판단(evidence_ids)
    또는 JSON 다운로드 쪽을 봐야 한다.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_df = source_tx.copy()
    if id_col not in csv_df.columns:
        # PaySim처럼 거래 고유 ID 컬럼이 없으면 행 인덱스를 그대로 쓴다
        # (render_details의 evidence_ids 해석과 동일한 규칙).
        csv_df.insert(0, id_col, csv_df.index.astype(str))
    csv_df["fraud_score"] = result["fraud_score"]
    csv_df["triggered_rules"] = ", ".join(result["triggered_rules"])
    lead_cols = [c for c in (id_col, entity_col) if c]
    csv_df = csv_df[lead_cols + [c for c in csv_df.columns if c not in lead_cols]]

    dc1, dc2 = st.columns(2)
    dc1.download_button(
        "결과 다운로드 (CSV)",
        data=csv_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"fds_result_{model_slug}_{timestamp}.csv",
        mime="text/csv",
    )
    dc2.download_button(
        "규칙별 상세 판단 다운로드 (JSON)",
        data=json.dumps(result["details"], ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"fds_result_{model_slug}_{timestamp}_details.json",
        mime="application/json",
    )


if mode == "파산관재인 FDS":
    from bankruptcy_fds.src.models import predict as bankruptcy_predict

    SAMPLE_PATH = ROOT / "bankruptcy_fds" / "data" / "sample_transactions.csv"

    uploaded = st.file_uploader("거래 CSV 업로드 (파산관재인 FDS 스키마)", type="csv")
    if uploaded is not None:
        tx = pd.read_csv(uploaded, parse_dates=["transaction_date"])
    elif SAMPLE_PATH.exists():
        st.caption(f"업로드된 파일이 없어 샘플 데이터를 사용합니다: {SAMPLE_PATH.name}")
        tx = pd.read_csv(SAMPLE_PATH, parse_dates=["transaction_date"])
    else:
        st.info("거래 CSV를 업로드하세요.")
        st.stop()

    debtor_ids = sorted(tx["debtor_id"].unique())
    selected = st.sidebar.selectbox("검토할 파산자", debtor_ids)
    debtor_tx = tx[tx["debtor_id"] == selected]

    filing_date_input = st.sidebar.date_input(
        "파산 신청일 (비우면 최신 거래일 사용)", value=None
    )
    filing_date = pd.Timestamp(filing_date_input) if filing_date_input else None

    result = bankruptcy_predict(debtor_tx, filing_date=filing_date)

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Score", f"{result['fraud_score']:.2f}")
    c2.metric("Triggered Rules", len(result["triggered_rules"]))
    c3.metric("전체 거래 건수", len(debtor_tx))

    st.subheader("규칙별 상세 판단")
    render_details(result["details"], debtor_tx, evidence_col="transaction_id")
    render_downloads(debtor_tx, result, model_slug="bankruptcy",
                      id_col="transaction_id", entity_col="debtor_id")

    st.subheader("전체 거래 내역")
    st.dataframe(debtor_tx, use_container_width=True)

else:  # 은행권 FDS
    from kaggle_bank_fds.src.models.predictor import predict as bank_predict

    # 저장소에 PaySim CSV가 포함되어 있지 않으므로(Kaggle에서 별도 다운로드),
    # 업로드가 없을 때는 이체→즉시현금화 패턴을 담은 최소 샘플 배치로 대체한다.
    SAMPLE_TX = pd.DataFrame([
        {
            "step": 1, "type": "TRANSFER", "amount": 100_000,
            "nameOrig": "C1", "oldbalanceOrg": 100_000, "newbalanceOrig": 0,
            "nameDest": "C2", "oldbalanceDest": 0, "newbalanceDest": 100_000,
        },
        {
            "step": 2, "type": "CASH_OUT", "amount": 99_000,
            "nameOrig": "C2", "oldbalanceOrg": 100_000, "newbalanceOrig": 1_000,
            "nameDest": "C3", "oldbalanceDest": 0, "newbalanceDest": 99_000,
        },
    ])

    uploaded = st.file_uploader("거래 CSV 업로드 (PaySim 스키마)", type="csv")
    if uploaded is not None:
        tx = pd.read_csv(uploaded)
    else:
        st.caption("업로드된 파일이 없어 내장 샘플 배치(이체→즉시현금화 2건)를 사용합니다.")
        tx = SAMPLE_TX

    st.sidebar.write(f"배치 거래 건수: {len(tx)}건")
    st.sidebar.caption("은행권 FDS는 계좌 단위가 아니라 배치 전체를 한 번에 평가합니다.")

    result = bank_predict(tx)

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Score", f"{result['fraud_score']:.2f}")
    c2.metric("Triggered Rules", len(result["triggered_rules"]))
    c3.metric("전체 거래 건수", len(tx))

    st.subheader("규칙별 상세 판단")
    # PaySim에는 거래 고유 ID가 없어 evidence_ids는 원본 DataFrame의 행 인덱스다.
    render_details(result["details"], tx, evidence_col=None)
    # PaySim에는 debtor_id 개념이 없다(계좌 여러 개가 섞인 배치 평가) —
    # entity_col을 생략하고 transaction_id만 행 인덱스로부터 만들어 붙인다.
    render_downloads(tx, result, model_slug="bank",
                      id_col="transaction_id", entity_col=None)

    st.subheader("전체 거래 내역")
    st.dataframe(tx, use_container_width=True)
