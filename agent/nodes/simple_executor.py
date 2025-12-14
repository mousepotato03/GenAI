"""
Simple Tool Executor Node - 단순 질문용 도구 실행
"""
import json
from typing import Dict

from langchain_core.messages import ToolMessage

from agent.state import AgentState
from tools.calculator import calculate_subscription_cost, check_tool_freshness, get_current_time
from tools.search import google_search_tool


# 단순 질문용 도구 맵
SIMPLE_TOOLS_MAP = {
    "get_current_time": get_current_time,
    "google_search_tool": google_search_tool,
    "calculate_subscription_cost": calculate_subscription_cost,
    "check_tool_freshness": check_tool_freshness
}


def simple_tool_executor(state: AgentState) -> Dict:
    """
    단순 질문용 도구 실행 노드

    simple_llm_node가 요청한 Tool Call을 실행합니다.
    """
    print("[Node] simple_tool_executor 실행")

    tool_result = state.get("tool_result")

    if not tool_result:
        print("  - tool_result 없음, 스킵")
        return {}

    try:
        tool_call = json.loads(tool_result)
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        tool_id = tool_call.get("id", "")

        print(f"  - 실행 도구: {tool_name}")
        print(f"  - 인자: {tool_args}")

        # 도구 실행
        tool_func = SIMPLE_TOOLS_MAP.get(tool_name)
        if tool_func:
            result = tool_func.invoke(tool_args)
            observation = str(result)
        else:
            observation = f"알 수 없는 도구: {tool_name}"

        print(f"  - 결과: {observation[:100]}...")

        # ToolMessage로 Observation 반환
        return {
            "messages": [ToolMessage(content=observation, tool_call_id=tool_id)],
            "tool_result": None  # 다음 루프를 위해 초기화
        }

    except Exception as e:
        print(f"  - 도구 실행 오류: {e}")
        return {
            "messages": [ToolMessage(content=f"도구 실행 오류: {str(e)}", tool_call_id="error")],
            "tool_result": None,
            "error": str(e)
        }
