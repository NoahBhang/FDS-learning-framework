-- ============================================================
-- FDS 데이터베이스 스키마 (schema.sql)
-- ============================================================
--
-- [설계 철학]
-- 데이터베이스는 "현실의 장기 기억"이다.
-- 파이썬 프로그램이 꺼졌다 켜져도, 사건의 기록은 여기 남는다.
--
-- 테이블 설계의 원칙은 단 하나다:
--   "하나의 테이블은 하나의 현실 개념을 담는다"
--
--   debtors            = 파산자라는 존재
--   transactions       = 거래라는 사건
--   detection_results  = 시스템의 판단이라는 기록
--
-- 이 세 테이블의 관계가 곧 FDS의 뼈대다:
--   존재(debtors) 1 ──── N 사건(transactions)
--   존재(debtors) 1 ──── N 판단(detection_results)
-- ============================================================


-- ------------------------------------------------------------
-- 1. 파산자 테이블: "존재"를 담는다
-- ------------------------------------------------------------
-- filing_date(파산 신청일)를 여기에 두는 이유:
--   신청일은 거래의 속성이 아니라 파산자의 속성이기 때문이다.
--   모든 의심 규칙은 "신청일로부터 며칠 전"을 기준으로 삼으므로,
--   이 값이 거래 테이블에 중복 저장되면 갱신 이상(update anomaly)이 생긴다.
--   → 정규화: 한 사실은 한 곳에만 기록한다.
CREATE TABLE IF NOT EXISTS debtors (
    debtor_id     TEXT PRIMARY KEY,      -- 파산자 고유 ID (예: D001)
    name          TEXT NOT NULL,          -- 성명
    filing_date   DATE NOT NULL           -- 파산 신청일 (모든 시간 규칙의 기준점)
);


-- ------------------------------------------------------------
-- 2. 거래 테이블: "사건"을 담는다
-- ------------------------------------------------------------
-- 거래는 불변(immutable)의 사실이다. 한번 일어난 송금은 수정되지 않는다.
-- 따라서 이 테이블에는 UPDATE가 없고 INSERT만 있다.
-- (파이썬 쪽에서 Transaction을 frozen dataclass로 만드는 이유와 정확히 같다)
--
-- relation_type을 거래에 두는 이유:
--   "김OO가 가족인가"는 수취인의 속성이지만, 실무에서 관재인은
--   거래 시점의 조사 결과로 관계를 기록한다. 프로토타입 단계에서는
--   거래에 붙여 두고, 고도화 시 receivers 테이블로 분리한다. (확장 지점)
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id    TEXT PRIMARY KEY,   -- 거래 고유 ID (예: T0001)
    debtor_id         TEXT NOT NULL,      -- 어느 파산자의 거래인가 (외래키)
    transaction_date  DATE NOT NULL,      -- 거래일
    amount            INTEGER NOT NULL,   -- 금액 (원 단위 정수. 부동소수점 오차 방지)
    sender            TEXT NOT NULL,      -- 보낸 주체 (파산자 본인 or 중간 계좌)
    receiver          TEXT NOT NULL,      -- 받은 주체
    relation_type     TEXT,               -- 가족 / 지인 / 법인 / 불명
    transaction_type  TEXT NOT NULL,      -- 송금 / 현금인출 / 카드결제
    FOREIGN KEY (debtor_id) REFERENCES debtors (debtor_id)
);

-- [인덱스 설계의 이유]
-- FDS의 모든 질문은 "이 파산자의, 이 기간의 거래"로 시작한다.
--   SELECT * FROM transactions
--   WHERE debtor_id = ? AND transaction_date >= ?
-- 이 질의가 인덱스 없이 실행되면 전체 테이블 스캔 O(n)이고,
-- 복합 인덱스가 있으면 O(log n)으로 줄어든다.
-- → 2부 확장편에서 다룬 "복잡도는 실무 사용 가능성을 가른다"의 실물이다.
CREATE INDEX IF NOT EXISTS idx_tx_debtor_date
    ON transactions (debtor_id, transaction_date);

-- 레이어링(자금 경로 추적)은 "누가 누구에게 보냈나"로 그래프를 만든다.
-- sender 기준 탐색이 잦으므로 별도 인덱스를 둔다.
CREATE INDEX IF NOT EXISTS idx_tx_sender
    ON transactions (sender);


-- ------------------------------------------------------------
-- 3. 판단 결과 테이블: "시스템의 판단"을 담는다
-- ------------------------------------------------------------
-- 판단을 저장하는 이유는 두 가지다.
--   (1) 감사 가능성: 관재인이 "그때 시스템이 왜 그렇게 판단했나"를
--       나중에 재확인할 수 있어야 한다. 법률 도메인에서는 필수다.
--   (2) 판단과 사실의 분리: 거래(사실)는 불변이지만 판단은
--       규칙이 개선되면 달라진다. 섞어 저장하면 안 된다.
--
-- evidence_ids를 JSON 문자열로 두는 이유:
--   판단 하나가 근거 거래 여러 건을 가리키는 1:N 관계다.
--   정석은 별도 매핑 테이블이지만, 프로토타입에서는 JSON으로 충분하고
--   "무엇이 근거였는가"를 잃지 않는 것이 핵심이다. (확장 지점)
CREATE TABLE IF NOT EXISTS detection_results (
    result_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    debtor_id     TEXT NOT NULL,          -- 누구에 대한 판단인가
    rule_name     TEXT NOT NULL,          -- 어떤 규칙의 판단인가
    risk_type     TEXT NOT NULL,          -- 의심 유형 (쪼개기 송금 의심 등)
    is_suspicious INTEGER NOT NULL,       -- 0 or 1 (SQLite에는 BOOLEAN이 없다)
    risk_score    INTEGER NOT NULL,       -- 이 규칙이 부여한 점수
    reason        TEXT NOT NULL,          -- 사람이 읽는 근거 문장 (설명 가능성의 핵심)
    evidence_ids  TEXT,                   -- 근거 거래 ID 목록 (JSON 배열 문자열)
    detected_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (debtor_id) REFERENCES debtors (debtor_id)
);
