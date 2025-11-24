"""
NODE-02: Human Review

계획을 사용자에게 제시하고 승인을 기다립니다.
LangGraph의 interrupt 기능을 활용합니다.
"""

from typing import Dict
from src.agents.state import AgentState


class ReviewNode:
    """계획 승인을 위한 Human-in-the-loop 노드"""

    def __call__(self, state: AgentState) -> Dict:
        """Reviewer 노드 실행

        이 노드는 LangGraph의 interrupt_before 설정과 함께 사용됩니다.
        실제로는 상태를 그대로 반환하며, interrupt가 발생합니다.

        사용자가 승인하면:
        - state["plan_approved"] = True로 설정되고 그래프 재개

        사용자가 거절하면:
        - state["user_feedback"]에 피드백 입력
        - state["plan_approved"] = False 유지
        - Planner 노드로 돌아가서 계획 재생성

        Args:
            state: 현재 그래프 상태 (plan 포함)

        Returns:
            업데이트된 상태 (변경 없음, interrupt 대기)
        """
        # 계획이 이미 승인되었는지 확인
        if state.get("plan_approved", False):
            print("[Reviewer] 계획이 승인되었습니다. 다음 단계로 진행합니다.")
            return {}

        # 계획을 사용자에게 표시
        plan = state.get("plan", [])
        if plan:
            print("\n[Reviewer] 다음 계획을 검토해 주세요:")
            for i, task in enumerate(plan, 1):
                print(f"{i}. {task['description']} (카테고리: {task['category']})")
            print("\n승인하시면 API의 /approve 엔드포인트로 요청해 주세요.")
        else:
            print("[Reviewer] 경고: 계획이 비어있습니다.")

        # 상태를 그대로 반환 (interrupt가 발생)
        # FastAPI에서 /approve 엔드포인트를 통해 그래프를 재개합니다
        return {}


def should_continue(state: AgentState) -> str:
    """조건부 엣지: 승인 여부에 따라 다음 노드 결정

    이 함수는 LangGraph의 add_conditional_edges에서 사용됩니다.

    Args:
        state: 현재 그래프 상태

    Returns:
        "approved": 승인됨, Research로 진행
        "revise": 거절됨, Plan으로 돌아가서 수정
    """
    if state.get("plan_approved", False):
        return "approved"
    else:
        return "revise"
