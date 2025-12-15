"""
Routing Functions - 그래프 분기 로직
"""
from agent.state import AgentState


def route_after_llm_router(state: AgentState) -> str:
    """llm_router 후 라우팅"""
    if state.get("is_complex_task", False):
        return "planning_node"
    else:
        return "simple_llm_node"  # 단순 질문 -> ReAct 패턴


def route_after_simple_llm(state: AgentState) -> str:
    """simple_llm_node 후 라우팅 (단순 질문 ReAct 루프)"""
    tool_result = state.get("tool_result")

    if tool_result is not None:
        return "simple_executor"
    else:
        # 최종 답변 완료 -> reflection으로
        return "reflection_node"


def route_after_recommend(state: AgentState) -> str:
    """ReAct 루프 분기 결정"""
    tool_result = state.get("tool_result")

    if tool_result is not None:
        return "tool_executor"
    else:
        # 모든 태스크 완료 확인
        sub_tasks = state.get("sub_tasks", [])
        current_idx = state.get("current_task_idx", 0)

        if current_idx < len(sub_tasks):
            # 아직 처리할 태스크가 남음 -> 계속 추천
            return "recommend_tool_node"
        else:
            # 모든 태스크 완료 -> 가이드 생성
            return "guide_generation_node"
