# Bank FDS Demo CSV

Streamlit uploader에 바로 사용할 수 있는 자체 생성 synthetic PaySim 예제입니다.

| 파일 | 목적 | 예상 점수 | Triggered Rules | Alert |
|---|---|---:|---|---|
| `clean.csv` | 정상 분석 | 0/100 | 없음 | 미생성 |
| `exact_overlap.csv` | Rapid/Split Evidence 완전 중복 | 35/100 | Rapid, Split | 생성 |
| `partial_overlap.csv` | Rapid/Split Evidence 부분 중복 | 65/100 | Rapid, Split | 생성 |
| `rounded_full_balance.csv` | FullBalance/Rounded 독립 중첩 | 40/100 | FullBalance, Rounded | 생성 |

저장소 root에서 결정적으로 재생성합니다.

```bash
python kaggle_bank_fds/scripts/generate_demo_data.py
```

Generator output과 tracked CSV는 byte 단위로 검증됩니다. 이 데이터에는 실제 개인정보가 없으며 점수는 사기 발생 확률이 아닙니다.
