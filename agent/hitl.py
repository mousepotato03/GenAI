"""
Human-in-the-Loop - 사용자 피드백 처리
"""
import json
from typing import Dict, List

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from agent.state import AgentState
from core.llm import get_llm
from prompts.intent import (
    INTENT_ANALYSIS_SYSTEM_PROMPT, INTENT_ANALYSIS_USER_TEMPLATE,
    MODIFY_PLAN_SYSTEM_PROMPT, MODIFY_PLAN_USER_TEMPLATE
)
from core.utils import extract_json # <-- 추가


def analyze_user_intent(user_text: str, plan_summary: str) -> dict:
    """사용자의 자연어 응답에서 의도를 분석"""
    llm = get_llm(temperature=0.1)

    user_prompt = INTENT_ANALYSIS_USER_TEMPLATE.format(
        plan_summary=plan_summary,
        user_response=user_text
    )

    response = llm.invoke([
        SystemMessage(content=INTENT_ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    try:
        # 개선 코드: extract_json을 사용하여 JSON 텍스트 추출
        response_text = extract_json(response.content)

        result = json.loads(response_text)
        return {
            "action": result.get("intent", "approve"),
            "feedback": result.get("feedback", "")
        }
    except Exception as e:
        print(f"의도 분석 파싱 오류: {e}")
        return {"action": "approve", "feedback": ""}


def modify_subtasks(current_subtasks: List[str], feedback: str) -> List[str]:
    """사용자 피드백을 반영하여 서브태스크 수정"""
    llm = get_llm(temperature=0.3)

    current_plan = "\n".join([f"- {task}" for task in current_subtasks])

    user_prompt = MODIFY_PLAN_USER_TEMPLATE.format(
        current_plan=current_plan,
        feedback=feedback
    )

    response = llm.invoke([
        SystemMessage(content=MODIFY_PLAN_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    try:
        # 개선 코드: extract_json을 사용하여 JSON 텍스트 추출
        response_text = extract_json(response.content)

        modified_tasks = json.loads(response_text)

        if isinstance(modified_tasks, list):
            # [{id, description, ...}] 형식이면 description만 추출
            if modified_tasks and isinstance(modified_tasks[0], dict):
                return [t.get("description", str(t)) for t in modified_tasks]
            return modified_tasks

        return current_subtasks
    except Exception as e:
        print(f"계획 수정 파싱 오류: {e}")
        return current_subtasks


def handle_human_feedback(state: AgentState, user_response: str) -> Dict:
    """
    사용자 피드백 처리 (interrupt 후 호출)

    Args:
        state: 현재 상태
        user_response: 사용자 응답 (자연어)

    Returns:
        업데이트할 상태 딕셔너리
    """
    sub_tasks = state.get("sub_tasks", [])
    plan_summary = "\n".join([f"{i+1}. {task}" for i, task in enumerate(sub_tasks)])

    # 의도 분석
    intent_result = analyze_user_intent(user_response, plan_summary)
    action = intent_result["action"]
    feedback = intent_result["feedback"]

    print(f"[Human Feedback] action={action}, feedback={feedback}")

    if action == "cancel":
        return {
            "final_guide": "작업이 취소되었습니다. 다른 질문이 있으시면 말씀해주세요.",
            "error": "사용자 취소"
        }
    elif action == "modify":
        # 계획 수정
        modified_tasks = modify_subtasks(sub_tasks, feedback)
        return {
            "sub_tasks": modified_tasks,
            "user_feedback": feedback,
            "current_task_idx": 0,
            "tool_recommendations": {},
            "task_completed": False,
            "messages": [AIMessage(content=f"계획이 수정되었습니다:\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(modified_tasks)]))]
        }
    else:
        # 승인
        return {
            "user_feedback": None,
            "messages": [AIMessage(content="계획이 승인되었습니다. 도구 추천을 시작합니다.")]
        }
