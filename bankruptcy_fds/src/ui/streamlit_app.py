"""
bankruptcy-fds/src/ui/streamlit_app.py
============================================================
관재인 대시보드 — "사용자 인터페이스 세계"

[이 화면의 설계 원칙: 판단이 아니라 검토를 돕는다]
FDS는 관재인을 대체하지 않는다. 시스템의 역할은
"어디를 먼저 볼 것인가"의 우선순위와 "왜 의심하는가"의 근거를
제공하는 것까지다. 최종 법률 판단은 사람이 한다.
그래서 화면의 모든 요소는 점수(우선순위) + 근거 문장 + 증거 거래
세트로 구성된다. 점수만 보여주는 화면은 이 도메인에서 실격이다.

실행: streamlit run bankruptcy-fds/src/ui/streamlit_app.py
(먼저 scripts/run_demo.py를 실행해 DB를 만들어 두어야 한다)
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "fds.db"

st.set_page_config(page_title="파산관재인 FDS", layout="wide")
st.title("파산관재인 FDS — 의심 거래 검토 대시보드")

if not DB_PATH.exists():
    st.error("DB가 없습니다. 먼저 `python scripts/run_demo.py`를 실행하세요.")
    st.stop()

conn = sqlite3.connect(DB_PATH)

# ── 좌측: 파산자 선택 ─────────────────────────────────────────
debtors = pd.read_sql("SELECT * FROM debtors", conn)
selected = st.sidebar.selectbox("검토할 파산자", debtors["debtor_id"])
info = debtors[debtors["debtor_id"] == selected].iloc[0]
st.sidebar.write(f"성명: {info['name']}")
st.sidebar.write(f"파산 신청일: {info['filing_date']}")

# ── 상단: 종합 판단 카드 ──────────────────────────────────────
results = pd.read_sql(
    "SELECT * FROM detection_results WHERE debtor_id = ? AND is_suspicious = 1",
    conn, params=(selected,),
)
total = min(100, int(results["risk_score"].sum()))
grade = "고위험" if total >= 70 else "중위험" if total >= 40 else \
        "저위험" if total >= 1 else "정상"

c1, c2, c3 = st.columns(3)
c1.metric("종합 위험 점수", f"{total}점")
c2.metric("등급", grade)
c3.metric("의심 유형 수", f"{len(results)}건")

# ── 중단: 의심 유형별 근거 (설명 가능성의 화면화) ────────────────
st.subheader("의심 정황과 근거")
if results.empty:
    st.success("발견된 의심 정황이 없습니다.")
for _, row in results.iterrows():
    with st.expander(f"⚠ {row['risk_type']}  (+{row['risk_score']}점)"):
        st.write(row["reason"])
        evidence_ids = json.loads(row["evidence_ids"] or "[]")
        if evidence_ids:
            ev = pd.read_sql(
                f"SELECT * FROM transactions WHERE transaction_id IN "
                f"({','.join(['?'] * len(evidence_ids))})",
                conn, params=evidence_ids,
            )
            st.dataframe(ev, use_container_width=True)

# ── 하단: 전체 거래 타임라인 ──────────────────────────────────
st.subheader("전체 거래 내역")
tx = pd.read_sql(
    "SELECT * FROM transactions WHERE debtor_id = ? ORDER BY transaction_date",
    conn, params=(selected,),
)
st.dataframe(tx, use_container_width=True)
