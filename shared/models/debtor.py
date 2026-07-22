"""
shared/models/debtor.py
============================================================
파산자(Debtor) — "존재"의 추상화

[Transaction과 달리 frozen이 아닌 이유]
거래는 불변의 사실이지만, 파산자는 조사가 진행되며
상태가 바뀌는 존재다 (위험 점수가 누적되고, 조사 단계가 진행된다).
불변으로 만들 것과 가변으로 둘 것을 구분하는 것 자체가
도메인 이해의 표현이다.

[filing_date가 왜 여기 있는가]
파산 신청일은 모든 시간 기반 규칙("신청 전 30일 이내")의 기준점이다.
이 값은 거래의 속성이 아니라 파산자의 속성이므로 여기에 둔다.
DB의 debtors 테이블과 1:1로 대응한다 — 코드의 클래스 구조와
DB의 테이블 구조가 같은 현실을 다른 매체에 담은 것임을 보여준다.
"""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Debtor:
    debtor_id: str            # 파산자 고유 ID
    name: str                 # 성명
    filing_date: date         # 파산 신청일 (모든 시간 규칙의 기준점)

    # 아래 두 필드는 "조사가 진행되며 변하는 상태"다.
    # 1부 문서 4장 '상태' 개념의 실물: 프로그램이 실행되며
    # 이 값들이 어떻게 변하는지가 곧 조사의 진행이다.
    risk_score: int = 0                          # 누적 위험 점수
    suspicious_flags: list = field(default_factory=list)  # 발견된 의심 유형 목록
