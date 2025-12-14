"""
Human Approval Node - Human-in-the-Loop 승인 대기
"""
from typing import Dict

from agent.state import AgentState


def human_approval_node(state: AgentState) -> Dict:
    """
    Human-in-the-Loop 승인 대기 노드

    planning_node에서 생성된 계획을 사용자가 승인할 때까지 대기합니다.
    interrupt_before가 이 노드에 설정되어 있어, 실제로 이 함수가 실행되기 전에 멈춥니다.
    승인 후 재개되면 단순히 통과합니다.
    """
    print("[Node] human_approval 실행 (승인 완료)")

    # 승인 후 다음 노드로 진행
    return {}
