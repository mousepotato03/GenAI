"""
Guide Generation Node - 최종 가이드 생성
"""
import json
from typing import Dict

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from core.llm import get_llm
from core.memory import get_memory_manager
from tools.search import hybrid_search
from prompts.guide import (
    GUIDE_GENERATION_SYSTEM_PROMPT, GUIDE_GENERATION_USER_TEMPLATE,
    GUIDE_SIMPLE_QA_SYSTEM_PROMPT, GUIDE_SIMPLE_QA_USER_TEMPLATE
)
from prompts.formatters import format_user_profile


def guide_generation_node(state: AgentState) -> Dict:
    """
    [작업 5] 최종 가이드 생성

    수집된 정보를 바탕으로 최종 워크플로우 가이드를 생성합니다.
    """
    print("[Node] guide_generation 실행")

    llm = get_llm(temperature=0.7)
    is_complex = state.get("is_complex_task", False)
    user_query = state.get("user_query", "")
    user_profile = state.get("user_profile")
    retrieved_docs = state.get("retrieved_docs", [])

    if not is_complex:
        # 단순 Q&A: 간단한 답변 생성
        print("  - 단순 Q&A 모드")

        # 검색 수행 (단순 Q&A에도 컨텍스트 필요)
        if not retrieved_docs:
            memory = get_memory_manager()
            results, _ = hybrid_search(
                memory_manager=memory,
                query=user_query,
                k=3,
                threshold=0.5,
                use_web_fallback=False
            )
            retrieved_docs = results

        user_prompt = GUIDE_SIMPLE_QA_USER_TEMPLATE.format(
            user_query=user_query,
            retrieved_docs=json.dumps(retrieved_docs[:5], ensure_ascii=False, indent=2) if retrieved_docs else "검색 결과 없음",
            user_profile=format_user_profile(user_profile)
        )

        response = llm.invoke([
            SystemMessage(content=GUIDE_SIMPLE_QA_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        return {
            "final_guide": response.content,
            "retrieved_docs": retrieved_docs,
            "messages": [response]
        }

    # 복잡한 작업: 상세 가이드 생성
    print("  - 복잡한 작업 모드")

    sub_tasks = state.get("sub_tasks", [])
    tool_recommendations = state.get("tool_recommendations", {})

    # 서브태스크 포맷팅
    sub_tasks_formatted = "\n".join([f"{i+1}. {task}" for i, task in enumerate(sub_tasks)])

    # 추천 도구 포맷팅
    recommendations_formatted = ""
    for task_id, recommendation in tool_recommendations.items():
        recommendations_formatted += f"\n### {task_id}\n{recommendation}\n"

    user_prompt = GUIDE_GENERATION_USER_TEMPLATE.format(
        user_query=user_query,
        sub_tasks=sub_tasks_formatted,
        tool_recommendations=recommendations_formatted if recommendations_formatted else "추천 정보 없음",
        retrieved_docs=json.dumps(retrieved_docs[:10], ensure_ascii=False, indent=2) if retrieved_docs else "검색 결과 없음",
        user_profile=format_user_profile(user_profile)
    )

    response = llm.invoke([
        SystemMessage(content=GUIDE_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    return {
        "final_guide": response.content,
        "messages": [response]
    }
