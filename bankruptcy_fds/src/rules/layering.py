"""
bankruptcy-fds/src/rules/layering.py
============================================================
LayeringRule — 레이어링(다단계 자금 이동) 탐지

[현실 문장]
"파산자의 돈이 중간 계좌를 한 번 이상 거쳐 최종 목적지로
이동했다면, 자금 추적을 어렵게 하려는 레이어링을 의심한다."
  예: 파산자 → 김OO → 김OO배우자   (2단계 경로)

[왜 딕셔너리로는 못 푸는가 — 자료구조의 한계가 드러나는 지점]
쪼개기 송금은 "파산자 → 수취인"이라는 1단계 관계였다.
딕셔너리 {수취인: 거래들}로 충분했다.
레이어링은 "누구에게서 누구에게로"가 연쇄되는 경로의 문제다.
경로는 딕셔너리가 아니라 그래프 G=(V, E)의 언어다.
  V(노드) = 계좌 주체들,  E(엣지) = 송금

이것이 2부 확장편에서 말한
"오일러가 1736년 다리 문제에서 만든 추상화가
 자금 흐름 추적에 그대로 재사용되는 지점"의 실제 코드다.

[알고리즘: BFS(너비 우선 탐색)]
파산자에서 출발해 송금 엣지를 따라 퍼져 나가며,
2단계 이상 떨어진 노드에 돈이 도달했는지 확인한다.
DFS가 아닌 BFS를 쓰는 이유: BFS는 가까운 경로부터 찾으므로
발견되는 경로가 곧 최단 이동 경로다. 관재인에게 보고할 때
"최소 몇 단계를 거쳤는가"가 그대로 나온다.

복잡도: O(V + E) — 노드와 엣지를 각각 한 번씩만 방문한다.
"""

from collections import deque

import pandas as pd

from shared.rules.base_fraud_rule import BaseFraudRule, RuleResult


class LayeringRule(BaseFraudRule):
    rule_name = "layering"
    risk_type = "레이어링(다단계 자금 이동) 의심"
    max_score = 40   # 은닉 의도가 가장 강한 수법이므로 최고 상한

    def __init__(self, min_hops: int = 2, min_amount: int = 1_000_000,
                 max_depth: int = 4):
        self.min_hops = min_hops        # 몇 단계 이상이면 의심하는가
        self.min_amount = min_amount    # 추적할 최소 금액 (소액 잡음 제거)
        self.max_depth = max_depth      # 탐색 깊이 상한 (무한 순환 방어)

    def evaluate(self, transactions_df: pd.DataFrame, **context) -> RuleResult:
        debtor_id = context["debtor_id"]

        # ── 1단계: 거래 테이블 → 그래프 변환 ────────────────────
        # 테이블(집계의 렌즈)을 그래프(경로의 렌즈)로 바꿔 낀다.
        # 인접 리스트: {보낸이: [(받은이, 금액, 거래ID), ...]}
        # 인접 행렬이 아닌 인접 리스트인 이유 — 자금 그래프는 희소하다.
        # 계좌 1만 개가 서로 모두 송금하지는 않으므로,
        # 존재하는 엣지만 저장하는 인접 리스트가 공간 효율적이다.
        transfers = transactions_df[
            (transactions_df["transaction_type"] == "송금")
            & (transactions_df["amount"] >= self.min_amount)
        ]
        adjacency: dict[str, list] = {}
        for row in transfers.itertuples():
            adjacency.setdefault(row.sender, []).append(
                (row.receiver, row.amount, row.transaction_id)
            )

        # ── 2단계: BFS 탐색 ─────────────────────────────────────
        # 큐에는 (현재 노드, 지금까지의 경로, 경로상 거래ID들)을 담는다.
        # 경로 자체를 들고 다니는 이유: 발견 즉시 그것이 곧 증거이기 때문.
        queue = deque([(debtor_id, [debtor_id], [])])
        visited = {debtor_id}           # 방문 집합: 순환(A→B→A) 무한루프 방어
        found_paths = []

        while queue:
            node, path, evidence = queue.popleft()

            if len(path) - 1 >= self.max_depth:   # 깊이 상한
                continue

            for next_node, amount, tx_id in adjacency.get(node, []):
                if next_node in visited:
                    continue
                visited.add(next_node)

                new_path = path + [next_node]
                new_evidence = evidence + [tx_id]

                # 파산자로부터 min_hops 단계 이상 떨어진 곳에 돈이 도달 → 적발
                if len(new_path) - 1 >= self.min_hops:
                    found_paths.append((new_path, new_evidence))

                queue.append((next_node, new_path, new_evidence))

        if not found_paths:
            return self._clean_result()

        # ── 3단계: 설명 생성 ────────────────────────────────────
        longest_path, evidence_ids = max(found_paths, key=lambda p: len(p[0]))
        path_str = " → ".join(longest_path)

        reason = (
            f"파산자의 자금이 중간 계좌를 거쳐 이동한 경로가 "
            f"{len(found_paths)}건 발견되었다. "
            f"최장 경로: {path_str} ({len(longest_path) - 1}단계). "
            f"자금 추적을 어렵게 하려는 레이어링 가능성이 있다."
        )

        # 경로가 길수록(은닉 단계가 많을수록) 높은 점수.
        score = min(self.max_score, 20 + 10 * (len(longest_path) - 2))

        return RuleResult(
            rule_name=self.rule_name,
            risk_type=self.risk_type,
            is_suspicious=True,
            risk_score=score,
            reason=reason,
            evidence_ids=evidence_ids,
        )
