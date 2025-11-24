"""
NODE-03: Research (Parallel Map)

각 서브태스크별로 AI 도구 후보를 검색합니다.
Hybrid Search (RAG + Web Search)를 사용하여 최대 5개 후보를 찾습니다.
"""

from typing import Dict, List
from src.agents.state import AgentState, ToolCandidate
from src.tools.search import HybridSearch


class ResearchNode:
    """AI 도구 검색을 담당하는 노드"""

    def __init__(self, hybrid_search: HybridSearch):
        """
        Args:
            hybrid_search: Hybrid Search 인스턴스
        """
        self.search = hybrid_search

    def __call__(self, state: AgentState) -> Dict:
        """Researcher 노드 실행

        각 서브태스크에 대해 병렬로 AI 도구를 검색합니다.

        Args:
            state: 현재 그래프 상태 (plan 포함)

        Returns:
            업데이트된 상태 (research_results, fallback_tasks 포함)
        """
        plan = state.get("plan", [])

        if not plan:
            print("[Researcher] 경고: 계획이 비어있습니다.")
            return {
                "research_results": {},
                "fallback_tasks": []
            }

        research_results = {}
        fallback_tasks = []

        print(f"\n[Researcher] {len(plan)}개 서브태스크에 대한 AI 도구 검색을 시작합니다...")

        # 각 서브태스크별로 검색 수행
        for i, subtask in enumerate(plan, 1):
            subtask_id = subtask["id"]
            description = subtask["description"]
            category = subtask["category"]

            print(f"\n[Researcher] [{i}/{len(plan)}] '{description}' 검색 중...")

            # Hybrid Search 실행
            try:
                candidates = self.search.search(
                    query=description,
                    category=category
                )

                # 결과를 ToolCandidate 형식으로 변환
                tool_candidates = []
                for candidate in candidates:
                    tool_candidates.append(ToolCandidate(
                        name=candidate.get("name", "Unknown"),
                        score=candidate.get("score", 0.0),
                        description=candidate.get("description", ""),
                        url=candidate.get("url", ""),
                        category=candidate.get("category", category),
                        pricing=candidate.get("pricing", "알 수 없음"),
                        features=candidate.get("features", []),
                        reputation_score=None,  # Evaluate 단계에서 추가
                        final_score=None,       # Evaluate 단계에서 추가
                        pros=None,
                        cons=None
                    ))

                research_results[subtask_id] = tool_candidates

                # Fallback 체크
                if self._check_fallback(tool_candidates):
                    fallback_tasks.append(subtask_id)
                    print(f"[Researcher] ⚠️  '{subtask_id}': 후보 점수가 낮아 Fallback 모드로 전환됩니다.")
                else:
                    print(f"[Researcher] ✅ '{subtask_id}': {len(tool_candidates)}개 후보 발견")

            except Exception as e:
                print(f"[Researcher] ❌ '{subtask_id}' 검색 실패: {e}")
                # 에러 발생 시 빈 결과 + Fallback
                research_results[subtask_id] = []
                fallback_tasks.append(subtask_id)

        print(f"\n[Researcher] 검색 완료! (Fallback: {len(fallback_tasks)}개)")

        return {
            "research_results": research_results,
            "fallback_tasks": fallback_tasks
        }

    def _check_fallback(self, candidates: List[ToolCandidate]) -> bool:
        """후보 점수가 모두 낮으면 Fallback 발동

        Args:
            candidates: 도구 후보 목록

        Returns:
            True: Fallback 필요
            False: 정상 진행
        """
        # 후보가 없으면 Fallback
        if not candidates:
            return True

        # 평균 점수가 0.5 미만이면 Fallback
        avg_score = sum(c["score"] for c in candidates) / len(candidates)
        return avg_score < 0.5
