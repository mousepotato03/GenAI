"""
NODE-04: Evaluate (Parallel Map)

후보 도구의 평판을 조회하고 최종 점수를 산정합니다.
점수 기준:
- Vector 유사도: 40%
- 평판 점수: 40%
- 접근성 (무료 여부): 20%
"""

from typing import Dict, List
from src.agents.state import AgentState, ToolCandidate
from src.tools.evaluator import ReputationEvaluator


class EvaluateNode:
    """도구 후보를 평가하는 노드"""

    def __init__(self, evaluator: ReputationEvaluator = None):
        """
        Args:
            evaluator: 평판 조회 도구 (None이면 자동 생성)
        """
        self.evaluator = evaluator or ReputationEvaluator()

    def __call__(self, state: AgentState) -> Dict:
        """Evaluator 노드 실행

        각 서브태스크의 후보 도구에 대해 평판 조회 및 점수 산정을 수행합니다.

        Args:
            state: 현재 그래프 상태 (research_results 포함)

        Returns:
            업데이트된 상태 (evaluation_results 포함)
        """
        research_results = state.get("research_results", {})

        if not research_results:
            print("[Evaluator] 경고: 검색 결과가 비어있습니다.")
            return {"evaluation_results": {}}

        evaluation_results = {}

        print(f"\n[Evaluator] {len(research_results)}개 서브태스크의 도구 평가를 시작합니다...")

        # 각 서브태스크별로 평가 수행
        for subtask_id, candidates in research_results.items():
            if not candidates:
                print(f"[Evaluator] '{subtask_id}': 후보 없음, 평가 건너뜀")
                evaluation_results[subtask_id] = []
                continue

            print(f"\n[Evaluator] '{subtask_id}': {len(candidates)}개 후보 평가 중...")

            # 각 후보에 대해 평판 조회
            scored_candidates = []
            for candidate in candidates:
                # 평판 점수 조회
                reputation_score = self.evaluator.get_reputation(
                    tool_name=candidate["name"]
                )

                # 최종 점수 계산
                final_score = self._calculate_score(candidate, reputation_score)

                # 후보 업데이트
                updated_candidate = ToolCandidate(
                    **candidate,
                    reputation_score=reputation_score,
                    final_score=final_score,
                    pros=[],  # TODO: LLM으로 장단점 생성 (선택사항)
                    cons=[]
                )

                scored_candidates.append(updated_candidate)

            # 점수 순 정렬
            scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

            # Top 3만 유지
            top_candidates = scored_candidates[:3]
            evaluation_results[subtask_id] = top_candidates

            print(f"[Evaluator] '{subtask_id}': Top 3 후보 선정 완료")
            for i, cand in enumerate(top_candidates, 1):
                print(f"  {i}. {cand['name']} (점수: {cand['final_score']:.2f})")

        print(f"\n[Evaluator] 평가 완료!")

        return {"evaluation_results": evaluation_results}

    def _calculate_score(self, candidate: ToolCandidate, reputation_score: float) -> float:
        """최종 점수 계산

        점수 구성:
        - Vector 유사도: 40%
        - 평판 점수: 40%
        - 접근성 (무료 여부): 20%

        Args:
            candidate: 도구 후보
            reputation_score: 평판 점수 (0~1)

        Returns:
            최종 점수 (0~1)
        """
        # 1. Vector 유사도 (40%)
        vector_score = candidate.get("score", 0.5) * 0.4

        # 2. 평판 점수 (40%)
        reputation_weight = reputation_score * 0.4

        # 3. 접근성 (20%)
        pricing = candidate.get("pricing", "알 수 없음").lower()
        if "무료" in pricing or "free" in pricing or "오픈소스" in pricing:
            accessibility = 1.0
        elif "프리미엄" in pricing or "premium" in pricing:
            accessibility = 0.7
        elif "유료" in pricing or "paid" in pricing:
            accessibility = 0.5
        else:
            accessibility = 0.6  # 알 수 없음

        accessibility_weight = accessibility * 0.2

        # 최종 점수
        final_score = vector_score + reputation_weight + accessibility_weight

        return round(final_score, 2)
