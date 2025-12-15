"""
Planning Node - 서브태스크 분해
"""
import json
from typing import Dict

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from agent.state import AgentState
from core.llm import get_llm
from prompts.planning import PLAN_SYSTEM_PROMPT, PLAN_USER_TEMPLATE
from prompts.formatters import format_user_profile
from core.utils import extract_json # <-- 추가


def planning_node(state: AgentState) -> Dict:
    """
    [작업 2] 서브태스크 분해

    복잡한 작업을 2-5개의 서브태스크로 분해합니다.
    """
    print("[Node] planning 실행")

    llm = get_llm(temperature=0.5)
    user_query = state.get("user_query", "")
    user_profile = state.get("user_profile")

    user_prompt = PLAN_USER_TEMPLATE.format(
        user_query=user_query,
        user_profile=format_user_profile(user_profile)
    )

    response = llm.invoke([
        SystemMessage(content=PLAN_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    try:
        # 개선 코드: extract_json을 사용하여 JSON 텍스트 추출
        response_text = extract_json(response.content)
        
        plan_data = json.loads(response_text)
        analysis = plan_data.get("analysis", "")
        subtasks_data = plan_data.get("subtasks", [])

        # sub_tasks: description 리스트로 변환
        sub_tasks = [t.get("description", "") for t in subtasks_data]

        print(f"  - 분석: {analysis}")
        print(f"  - 서브태스크 {len(sub_tasks)}개 생성")
        for i, task in enumerate(sub_tasks, 1):
            print(f"    {i}. {task}")

    except Exception as e:
        print(f"  - 계획 파싱 오류: {e}")
        analysis = "계획 수립 중 오류 발생"
        sub_tasks = [user_query]  # 원본 쿼리를 단일 태스크로

    # 계획 요약 메시지 생성
    plan_summary = "\n".join([f"{i+1}. {task}" for i, task in enumerate(sub_tasks)])
    plan_message = f"""## 작업 계획

**분석**: {analysis}

**수립된 계획**:
{plan_summary}

이 계획대로 진행할까요? (승인/수정/취소)"""

    return {
        "plan_analysis": analysis,
        "sub_tasks": sub_tasks,
        "tool_recommendations": {},
        "current_task_idx": 0,
        "tool_call_count": 0,
        "task_completed": False,
        "messages": [AIMessage(content=plan_message)]
    }
