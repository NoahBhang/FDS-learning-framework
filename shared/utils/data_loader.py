"""
shared/utils/data_loader.py
============================================================
DataLoader — 데이터베이스와 파이썬 사이의 다리

[DB ↔ 파이썬 연결의 3층 원리]

  1층: DB (장기 기억)
       프로그램이 꺼져도 남는 기록. SQL이 지배하는 세계.
  2층: DataFrame (작업 기억)
       분석하는 동안만 메모리에 존재. pandas가 지배하는 세계.
  3층: 객체 (판단 단위)
       Transaction, RuleResult 등. 도메인 언어가 지배하는 세계.

  연결 도구:
    1층 → 2층 : pandas.read_sql (SQL 결과가 곧 DataFrame이 된다)
    2층 → 1층 : DataFrame.to_sql (판단 결과를 다시 장기 기억에 남긴다)
    엔진      : SQLAlchemy engine — "어느 DB에 어떻게 접속하는가"를
                문자열 하나(접속 URL)로 추상화한다.
                sqlite:///fds.db → 파일 DB
                postgresql://user:pw@host/db → 서버 DB
                접속 URL만 바꾸면 코드 전체가 SQLite에서 PostgreSQL로
                이사한다. 이것이 인터페이스 추상화의 힘이다.

[이 클래스가 존재하는 이유]
규칙이나 파이프라인 코드에 SQL 문자열이 흩어져 있으면,
테이블 구조가 바뀔 때 온 코드를 뒤져야 한다.
"데이터를 가져오고 내보내는 일"을 이 한 파일에 가두면
DB 변경의 충격이 이 파일 안에서 멈춘다. → 책임의 분리.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


class DataLoader:
    def __init__(self, db_url: str):
        """
        db_url 예시:
          "sqlite:///bankruptcy-fds/data/fds.db"   (프로토타입)
          "postgresql://user:pw@localhost/fds"     (실서비스, URL만 교체)
        """
        self.engine = create_engine(db_url)

    # ----- 초기화 -------------------------------------------------

    def init_schema(self, schema_path: str) -> None:
        """schema.sql을 실행하여 테이블을 만든다. 멱등(idempotent)하다:
        IF NOT EXISTS 덕분에 몇 번을 실행해도 안전하다."""
        sql = Path(schema_path).read_text(encoding="utf-8")
        with self.engine.begin() as conn:          # begin() = 트랜잭션 단위 실행
            for statement in sql.split(";"):
                if statement.strip():
                    conn.execute(text(statement))

    def load_csv_to_table(self, csv_path: str, table: str) -> int:
        """CSV(외부 세계) → DB 테이블(장기 기억) 적재.

        if_exists="append": 기존 기록을 지우지 않는다.
        거래는 불변의 사실이므로 덮어쓰기(replace)가 아니라 추가만 한다."""
        df = pd.read_csv(csv_path)
        df.to_sql(table, self.engine, if_exists="append", index=False)
        return len(df)

    # ----- 읽기: 1층(DB) → 2층(DataFrame) --------------------------

    def get_debtors(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM debtors", self.engine,
                           parse_dates=["filing_date"])

    def get_transactions(self, debtor_id: str) -> pd.DataFrame:
        """
        한 파산자의 전체 거래를 가져온다.

        [파라미터 바인딩(:debtor_id)을 쓰는 이유]
        f-string으로 SQL을 조립하면 SQL 인젝션에 뚫린다.
        FDS는 공격자가 존재한다고 가정하는 시스템이므로
        보안 습관이 곧 도메인 요구사항이다.

        이 질의는 schema.sql의 idx_tx_debtor_date 인덱스를 타므로
        거래가 100만 건이어도 해당 파산자의 것만 O(log n)으로 찾는다.
        """
        return pd.read_sql(
            text("SELECT * FROM transactions WHERE debtor_id = :debtor_id"),
            self.engine,
            params={"debtor_id": debtor_id},
            parse_dates=["transaction_date"],
        )

    # ----- 쓰기: 판단 결과 → 1층(DB) --------------------------------

    def save_results(self, results_df: pd.DataFrame) -> None:
        """판단 결과를 detection_results 테이블에 남긴다.
        판단의 기록이 곧 감사 가능성(auditability)이다."""
        results_df.to_sql("detection_results", self.engine,
                          if_exists="append", index=False)
