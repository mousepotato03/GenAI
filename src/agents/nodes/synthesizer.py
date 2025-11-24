"""
NODE-06: Synthesize (Reduce)

모든 서브태스크의 결과를 하나의 일관된 Markdown 응답으로 통합합니다.
"""

from typing import Dict
from langchain_openai import ChatOpenAI
from src.agents.state import AgentState


class SynthesizeNode:
    """최종 응답 통합 노드"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Args:
            model: 사용할 LLM 모델명
            temperature: 생성 다양성 (0.0~1.0)
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def __call__(self, state: AgentState) -> Dict:
        """Synthesizer 노드 실행

        모든 결과를 통합하여 최종 Markdown 응답을 생성합니다.

        Args:
            state: 현재 그래프 상태 (plan, evaluation_results, guides 포함)

        Returns:
            업데이트된 상태 (final_response 포함)
        """
        print("\n[Synthesizer] 최종 응답 생성 중...")

        final_response = self._build_response(state)

        print("[Synthesizer] 최종 응답 생성 완료!")

        return {"final_response": final_response}

    def _build_response(self, state: AgentState) -> str:
        """Markdown 포맷 응답 생성

        구조:
        1. 서론: 사용자 목표 요약
        2. 단계별 가이드:
           - 각 서브태스크별로
           - 추천 도구 + 사용법
        3. 결론: 전체 워크플로우 요약

        Args:
            state: 현재 상태

        Returns:
            Markdown 형식의 최종 응답
        """
        user_query = state.get("user_query", "사용자 요청")
        plan = state.get("plan", [])
        evaluation_results = state.get("evaluation_results", {})
        guides = state.get("guides", {})
        fallback_tasks = state.get("fallback_tasks", [])

        sections = []

        # ===== 서론 =====
        intro = f"# {user_query} - 실행 가이드\n\n"
        intro += f"총 **{len(plan)}단계**로 나누어 진행합니다.\n\n"
        intro += "---\n\n"
        sections.append(intro)

        # ===== 본론: 단계별 가이드 =====
        for i, subtask in enumerate(plan, 1):
            subtask_id = subtask["id"]
            description = subtask["description"]
            category = subtask["category"]

            section = f"## 📍 단계 {i}: {description}\n\n"
            section += f"**카테고리**: {category}\n\n"

            if subtask_id in fallback_tasks:
                # Fallback 모드
                section += "### 💡 추천 방법: 범용 LLM 사용\n\n"
                section += "특화 도구를 찾지 못했습니다. 아래 방법을 사용하세요:\n\n"
                section += guides.get(subtask_id, "가이드를 생성할 수 없습니다.")
            else:
                # 일반 모드
                candidates = evaluation_results.get(subtask_id, [])
                if candidates:
                    top_tool = candidates[0]
                    section += f"### 🔧 추천 도구: {top_tool['name']}\n\n"

                    # 도구 정보
                    section += f"**공식 사이트**: {top_tool.get('url', 'N/A')}\n\n"
                    section += f"**가격**: {top_tool.get('pricing', '알 수 없음')}\n\n"
                    section += f"**평가 점수**: {top_tool.get('final_score', 0):.2f}/1.00\n\n"

                    # 상세 점수
                    section += "<details>\n"
                    section += "<summary>점수 세부 정보</summary>\n\n"
                    section += f"- Vector 유사도: {top_tool.get('score', 0):.2f}\n"
                    section += f"- 평판 점수: {top_tool.get('reputation_score', 0):.2f}\n"
                    section += f"- 접근성: {top_tool.get('pricing', '알 수 없음')}\n"
                    section += "</details>\n\n"

                    # 다른 후보들
                    if len(candidates) > 1:
                        section += "**다른 후보들**:\n"
                        for j, cand in enumerate(candidates[1:], 2):
                            section += f"{j}. {cand['name']} (점수: {cand.get('final_score', 0):.2f})\n"
                        section += "\n"

                    # 사용 가이드
                    section += "### 📖 사용 가이드\n\n"
                    section += guides.get(subtask_id, "가이드를 생성할 수 없습니다.")
                else:
                    # 후보 없음
                    section += "### ⚠️ 추천 도구를 찾을 수 없습니다\n\n"
                    section += guides.get(subtask_id, "범용 LLM을 사용하세요.")

            section += "\n\n---\n\n"
            sections.append(section)

        # ===== 결론 =====
        conclusion = "## 🎯 마무리\n\n"
        conclusion += f"위 {len(plan)}단계를 순서대로 따라하시면 **'{user_query}'** 목표를 달성하실 수 있습니다!\n\n"

        # 팁 추가
        conclusion += "### 💡 추가 팁\n\n"
        conclusion += "- 각 단계를 완료한 후 다음 단계로 넘어가세요\n"
        conclusion += "- 도구 사용 중 문제가 생기면 공식 문서를 참고하세요\n"
        conclusion += "- 무료 도구부터 시작해보는 것을 추천합니다\n"

        if fallback_tasks:
            conclusion += f"\n- 일부 작업({len(fallback_tasks)}개)은 범용 LLM으로 수행하세요\n"

        sections.append(conclusion)

        # 최종 결합
        return "".join(sections)
