"""
Tool Executor Node - 도구 실행
"""
import json
from typing import Dict

from langchain_core.messages import ToolMessage

from agent.state import AgentState
from tools.registry import execute_tool


def tool_executor_node(state: AgentState) -> Dict:
    """
    도구 실행 노드

    recommend_tool_node가 요청한 Tool Call을 실제로 실행합니다.
    """
    print("[Node] tool_executor 실행")

    tool_result = state.get("tool_result")
    retrieved_docs = state.get("retrieved_docs", [])

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
        result = execute_tool(tool_name, tool_args)
        observation = str(result)

        print(f"  - 결과 길이: {len(observation)} chars")

        # retrieved_docs 업데이트 (검색 결과인 경우)
        if tool_name in ["retrieve_docs", "google_search_tool"]:
            try:
                result_data = json.loads(observation)
                docs = result_data.get("results", [])
                if isinstance(docs, list):
                    retrieved_docs.extend(docs)
                    print(f"  - retrieved_docs에 {len(docs)}개 추가")
            except:
                pass

        # ToolMessage로 Observation 반환
        return {
            "retrieved_docs": retrieved_docs,
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
