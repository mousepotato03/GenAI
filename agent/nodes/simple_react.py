"""
Simple ReAct Node - 단순 질문용 ReAct 에이전트
"""
import json
import uuid
from typing import Dict

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from core.llm import get_llm
from core.config import MAX_TOOL_CALLS_PER_TASK
from tools.calculator import calculate_subscription_cost, check_tool_freshness, get_current_time
from tools.search import google_search_tool
from prompts.formatters import format_user_profile


SIMPLE_REACT_SYSTEM_PROMPT = """당신은 AI 도구 관련 질문에 답변하는 전문가입니다.

사용자의 간단한 질문에 친절하고 정확하게 답변해주세요.
필요한 경우 제공된 도구를 활용하여 정보를 수집하세요.

## 사용 가능한 도구:
1. get_current_time: 현재 날짜와 시간을 반환합니다.
2. google_search_tool: Google에서 최신 정보를 검색합니다.
3. calculate_subscription_cost: AI 도구들의 월간 구독료를 계산합니다.
4. check_tool_freshness: AI 도구 정보의 최신성을 확인합니다.

## 응답 지침:
- 최신 정보가 필요한 질문은 먼저 get_current_time으로 현재 날짜를 확인한 후, google_search_tool로 검색하세요.
- 도구가 필요한 경우에만 도구를 호출하세요.
- 도구 호출 결과를 바탕으로 최종 답변을 작성하세요.
- 한국어로 친절하게 답변하세요.
"""


def simple_llm_node(state: AgentState) -> Dict:
    """
    단순 질문용 ReAct Agent

    계산기, 시간 관련 도구를 사용하여 간단한 질문에 답변합니다.
    """
    print("[Node] simple_llm_node 실행")

    user_query = state.get("user_query", "")
    user_profile = state.get("user_profile")
    simple_tool_count = state.get("simple_tool_count", 0)
    messages = state.get("messages", [])

    print(f"  - 질문: {user_query}")
    print(f"  - 도구 호출 횟수: {simple_tool_count}")

    # 무한 루프 방지
    if simple_tool_count >= MAX_TOOL_CALLS_PER_TASK:
        print(f"  - 최대 도구 호출 횟수 도달 ({MAX_TOOL_CALLS_PER_TASK}), 최종 답변 생성")

        llm = get_llm(temperature=0.5)

        # 대화 히스토리를 바탕으로 최종 답변 생성
        final_response = llm.invoke([
            SystemMessage(content=SIMPLE_REACT_SYSTEM_PROMPT),
            *messages,
            HumanMessage(content="지금까지의 정보를 바탕으로 사용자 질문에 대한 최종 답변을 작성해주세요.")
        ])

        return {
            "tool_result": None,
            "final_answer": final_response.content,
            "messages": [final_response]
        }

    # 단순 질문용 도구 리스트
    simple_tools = [
        get_current_time,
        google_search_tool,
        calculate_subscription_cost,
        check_tool_freshness
    ]

    # LLM에 도구 바인딩
    llm_with_tools = get_llm(temperature=0.5).bind_tools(simple_tools)

    # 프로필 정보 포함
    profile_str = format_user_profile(user_profile) if user_profile else "정보 없음"

    user_prompt = f"""사용자 질문: {user_query}

사용자 프로필:
{profile_str}

위 질문에 대해 답변해주세요. 필요한 경우 도구를 사용하세요."""

    # 기존 메시지 히스토리 활용
    invoke_messages = [SystemMessage(content=SIMPLE_REACT_SYSTEM_PROMPT)]

    # 이전 메시지가 있으면 추가 (Tool 결과 포함)
    if len(messages) > 1:  # 첫 HumanMessage 외에 추가 메시지가 있으면
        invoke_messages.extend(messages[1:])  # 첫 메시지 제외하고 추가
        invoke_messages.append(HumanMessage(content="이전 도구 결과를 바탕으로 계속 답변해주세요."))
    else:
        invoke_messages.append(HumanMessage(content=user_prompt))

    response = llm_with_tools.invoke(invoke_messages)

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
            "simple_tool_count": simple_tool_count + 1,
            "messages": [response]
        }
    else:
        # Tool Call 없이 답변 완료
        print(f"  - 최종 답변 생성 완료")

        return {
            "tool_result": None,
            "final_answer": response.content,
            "messages": [response]
        }
