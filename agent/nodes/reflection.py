"""
Reflection Node - 메모리 저장
"""
import json
from typing import Dict

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from core.llm import get_llm
from core.memory import get_memory_manager
from prompts.reflection import MEMORY_EXTRACTOR_SYSTEM_PROMPT, MEMORY_EXTRACTOR_USER_TEMPLATE
from core.utils import extract_json # <-- 추가


def reflection_node(state: AgentState) -> Dict:
    """
    [작업 6] Reflection - 메모리 저장

    대화 내용을 분석하여 사용자 선호도를 추출하고 장기 메모리에 저장합니다.
    """
    print("[Node] reflection 실행")

    memory = get_memory_manager()
    user_id = state.get("user_id", "default_user")
    user_query = state.get("user_query", "")
    user_profile = state.get("user_profile")
    final_guide = state.get("final_guide", "")

    # 대화 내용 구성
    conversation = f"사용자: {user_query}\n\n에이전트: {final_guide}"

    llm = get_llm(temperature=0.3)

    user_prompt = MEMORY_EXTRACTOR_USER_TEMPLATE.format(
        conversation=conversation,
        existing_profile=json.dumps(user_profile, ensure_ascii=False, indent=2) if user_profile else "없음"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=MEMORY_EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        # 개선 코드: extract_json을 사용하여 JSON 텍스트 추출
        response_text = extract_json(response.content)

        new_preferences = json.loads(response_text)

        # 기존 프로필과 병합
        if user_profile:
            merged_profile = user_profile.copy()
            for key, value in new_preferences.items():
                if isinstance(value, list) and isinstance(merged_profile.get(key), list):
                    # 리스트는 합집합
                    merged_profile[key] = list(set(merged_profile[key] + value))
                elif value:  # 빈 값이 아니면 업데이트
                    merged_profile[key] = value
            new_preferences = merged_profile

        # 저장
        success = memory.save_user_profile(user_id, new_preferences)
        if success:
            print(f"  - 사용자 프로필 저장 완료: {user_id}")
        else:
            print(f"  - 사용자 프로필 저장 실패")

    except Exception as e:
        print(f"  - Reflection 오류: {e}")

    return {}
