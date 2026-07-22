"""
shared/models/transaction.py
============================================================
거래(Transaction) — "사건"의 추상화

[왜 dataclass인가]
거래는 행동하지 않는다. 거래는 일어난 사실의 기록일 뿐이다.
행동(메서드)이 필요 없는 순수 데이터 덩어리에는 class보다
dataclass가 정확한 표현이다. 코드가 곧 문서가 된다.

[왜 frozen=True인가]
2026년 1월 3일에 일어난 200만 원 송금은 영원히 그 사실이다.
프로그램 어딘가에서 실수로 tx.amount = 0 같은 변경이 일어나면
FDS 전체의 판단 근거가 오염된다. frozen=True는 그런 변경 시도를
그 자리에서 에러로 만든다. → "거래는 불변의 사실"이라는
도메인 규칙을 파이썬 문법으로 강제한 것이다.
DB 스키마에서 transactions 테이블에 UPDATE가 없는 것과 같은 철학이다.

[amount가 int인 이유]
금액을 float로 다루면 0.1 + 0.2 != 0.3 같은 부동소수점 오차가
법률 문서에 들어갈 수 있다. 원 단위 정수가 안전하다.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Transaction:
    transaction_id: str      # 거래 고유 ID
    debtor_id: str           # 어느 파산자의 거래인가
    transaction_date: date   # 거래일
    amount: int              # 금액 (원 단위 정수)
    sender: str              # 보낸 주체 (파산자 본인 또는 중간 계좌)
    receiver: str            # 받은 주체
    relation_type: str       # 가족 / 지인 / 법인 / 불명
    transaction_type: str    # 송금 / 현금인출 / 카드결제
