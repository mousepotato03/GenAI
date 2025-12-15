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
from tools.calculator import calculate_subscription_cost, check_tool_freshness, get_current_time, calculate_math
from tools.search import google_search_tool
from prompts.formatters import format_user_profile


SIMPLE_REACT_SYSTEM_PROMPT = """당신은 사용자를 돕는 친절한 AI 어시스턴트입니다.

사용자의 질문에 자연스럽고 도움이 되는 답변을 제공하세요.
필요한 경우 도구를 활용하여 정확한 정보를 제공할 수 있습니다.

## 사용 가능한 도구:
1. get_current_time: 현재 날짜와 시간 확인
2. google_search_tool: 최신 정보 검색
3. calculate_math: 간단한 수학 계산 (예: "22 * 34", "100 + 50")
4. calculate_subscription_cost: AI 도구 구독료 계산
5. check_tool_freshness: AI 도구 정보 최신성 확인

## 응답 원칙:
- 대화형 질문(인사, 잡담 등)에는 자연스럽게 응답하세요
- 최신 정보가 필요하면 도구를 활용하세요
- **중요: 사용자가 "A를 알려주고 B를 알려줘" 같은 복합 질문을 했다면:**
  1. 먼저 필요한 도구를 하나씩 사용하여 정보를 수집하세요
  2. 모든 정보를 수집한 후에 최종 답변을 작성하세요
  3. 일부 정보만 수집하고 답변하지 마세요
- 불필요한 도구 호출은 피하세요
- 항상 친절하고 명확하게 한국어로 답변하세요
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
        calculate_math,
        calculate_subscription_cost,
        check_tool_freshness
    ]

    # LLM에 도구 바인딩 (parallel tool calling 비활성화)
    llm_with_tools = get_llm(temperature=0.5).bind_tools(
        simple_tools,
        parallel_tool_calls=False  # 한 번에 하나의 도구만 호출
    )

    # 프로필 정보 포함
    profile_str = format_user_profile(user_profile) if user_profile else "정보 없음"

    # LLM invoke용 메시지 구성
    invoke_messages = [SystemMessage(content=SIMPLE_REACT_SYSTEM_PROMPT)]

    # 이전 대화 히스토리 추가 (단기 메모리) - 마지막 메시지(현재 질문) 제외
    from langchain_core.messages import AIMessage, ToolMessage
    history_messages = messages[:-1] if messages else []  # 마지막 HumanMessage 제외
    for msg in history_messages:
        # HumanMessage, AIMessage (tool_calls 없는 일반 응답) 포함
        if isinstance(msg, HumanMessage):
            invoke_messages.append(msg)
        elif isinstance(msg, AIMessage) and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
            invoke_messages.append(msg)

    # 첫 요청인 경우
    if simple_tool_count == 0:
        user_prompt = f"""사용자 질문: {user_query}

사용자 프로필:
{profile_str}

위 질문에 대해 답변해주세요. 필요한 경우 도구를 사용하세요."""
        invoke_messages.append(HumanMessage(content=user_prompt))
    else:
        # ReAct 루프 중: simple_react 관련 메시지만 필터링
        # - AIMessage with tool_calls (simple_react의 도구 호출)
        # - ToolMessage (도구 실행 결과)
        # llm_router의 AIMessage는 제외
        from langchain_core.messages import AIMessage, ToolMessage
        
        for msg in messages:
            # AIMessage with tool_calls만 포함 (llm_router 응답 제외)
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                invoke_messages.append(msg)
            # ToolMessage 포함
            elif isinstance(msg, ToolMessage):
                invoke_messages.append(msg)
        
        # 원래 사용자 질문을 상기시킴 (중요!)
        invoke_messages.append(HumanMessage(
            content=f"""[원래 사용자 질문]
{user_query}

위 질문의 모든 부분에 대해 필요한 정보를 수집했는지 확인하세요.
- 아직 답변하지 않은 부분이 있다면, 필요한 도구를 사용하세요.
- 모든 정보를 수집했다면, 최종 답변을 작성하세요."""
        ))

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
