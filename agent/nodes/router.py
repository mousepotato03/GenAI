"""
LLM Router Node - 질문 유형 분류
"""
import json
from typing import Dict

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from agent.state import AgentState
from core.llm import get_llm
from core.memory import get_memory_manager
from prompts.router import LLM_ROUTER_SYSTEM_PROMPT, LLM_ROUTER_USER_TEMPLATE
from prompts.formatters import format_user_profile
from core.utils import extract_json # <-- 수정: parse_json_safely 대신 extract_json 사용


def llm_router_node(state: AgentState) -> Dict:
    """
    [작업 1] 질문 유형 분류 - Entry Point

    사용자 질문이 단순 Q&A인지 복잡한 작업인지 판단합니다.
    """
    print("[Node] llm_router 실행")

    llm = get_llm(temperature=0.1)
    user_query = state.get("user_query", "")

    # 사용자 프로필 로드
    memory = get_memory_manager()
    user_id = state.get("user_id", "default_user")
    user_profile = memory.load_user_profile(user_id)

    user_prompt = LLM_ROUTER_USER_TEMPLATE.format(
        user_query=user_query,
        user_profile=format_user_profile(user_profile)
    )

    response = llm.invoke([
        SystemMessage(content=LLM_ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    try:
        # 개선 코드: extract_json을 사용하여 JSON 텍스트 추출 후, json.loads로 로드
        response_text = extract_json(response.content)
        result = json.loads(response_text)
        is_complex = result.get("is_complex", True)
        reason = result.get("reason", "")
    except Exception as e:
        print(f"  - 분류 파싱 오류: {e}, 기본값: 복잡한 작업")
        is_complex = True

    print(f"  - 분류 결과: {'복잡한 작업' if is_complex else '단순 Q&A'}")

    # messages는 반환하지 않음 (add_messages가 기존 messages를 유지)
    return {
        "is_complex_task": is_complex,
        "user_profile": user_profile
    }
