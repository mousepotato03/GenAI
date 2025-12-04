"""
Recommend Tool Node - ReAct 에이전트 도구 추천
"""
import json
import uuid
from typing import Dict

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from core.llm import get_llm
from core.config import MAX_TOOL_CALLS_PER_TASK
from tools.registry import get_all_tools
from prompts.recommend import RECOMMEND_TOOL_SYSTEM_PROMPT, RECOMMEND_TOOL_USER_TEMPLATE
from prompts.formatters import format_user_profile


def recommend_tool_node(state: AgentState) -> Dict:
    """
    [작업 3, 4] ReAct Agent - 도구 추천

    각 서브태스크에 대해 도구를 호출하여 최적의 AI 도구를 추천합니다.
    """
    print("[Node] recommend_tool 실행")

    sub_tasks = state.get("sub_tasks", [])
    current_idx = state.get("current_task_idx", 0)
    tool_call_count = state.get("tool_call_count", 0)
    tool_recommendations = state.get("tool_recommendations", {})
    retrieved_docs = state.get("retrieved_docs", [])
    user_profile = state.get("user_profile")

    # 모든 태스크 완료 체크
    if current_idx >= len(sub_tasks):
        print("  - 모든 서브태스크 처리 완료")
        return {"tool_result": None}

    current_task = sub_tasks[current_idx]
    print(f"  - 현재 태스크: {current_task} ({current_idx + 1}/{len(sub_tasks)})")

    # 무한 루프 방지
    if tool_call_count >= MAX_TOOL_CALLS_PER_TASK:
        print(f"  - 최대 도구 호출 횟수 도달 ({MAX_TOOL_CALLS_PER_TASK}), 다음 태스크로 이동")
        task_id = f"task_{current_idx + 1}"
        tool_recommendations[task_id] = "도구 검색 결과를 바탕으로 직접 확인이 필요합니다."
        return {
            "tool_result": None,
            "tool_recommendations": tool_recommendations,
            "current_task_idx": current_idx + 1,
            "tool_call_count": 0
        }

    # LLM에 도구 바인딩
    tools = get_all_tools()
    llm_with_tools = get_llm(temperature=0.3).bind_tools(tools)

    # 이전 검색 결과 컨텍스트
    previous_results = ""
    if retrieved_docs:
        recent_docs = retrieved_docs[-5:]  # 최근 5개
        previous_results = json.dumps(recent_docs, ensure_ascii=False, indent=2)

    user_prompt = RECOMMEND_TOOL_USER_TEMPLATE.format(
        current_task=current_task,
        previous_results=previous_results if previous_results else "없음",
        user_profile=format_user_profile(user_profile)
    )

    response = llm_with_tools.invoke([
        SystemMessage(content=RECOMMEND_TOOL_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    # Tool Call 확인
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"  - Tool Call: {tool_call['name']}")
        print(f"  - Args: {tool_call['args']}")

        return {
            "tool_result": json.dumps({
                "id": tool_call.get("id", str(uuid.uuid4())),
                "name": tool_call["name"],
                "arguments": tool_call["args"]
            }),
            "tool_call_count": tool_call_count + 1,
            "messages": [response]
        }
    else:
        # Tool Call 없이 추천 완료 -> 다음 태스크로
        print(f"  - 추천 완료, 다음 태스크로 이동")
        task_id = f"task_{current_idx + 1}"
        tool_recommendations[task_id] = response.content

        return {
            "tool_result": None,
            "tool_recommendations": tool_recommendations,
            "current_task_idx": current_idx + 1,
            "tool_call_count": 0,
            "messages": [response]
        }
