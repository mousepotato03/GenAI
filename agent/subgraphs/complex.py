"""
Complex Subgraph - 복잡 질문 처리 서브그래프
"""
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    planning_node,
    human_approval_node,
    recommend_tool_node,
    tool_executor_node,
    guide_generation_node
)


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


def create_complex_subgraph(checkpointer=None):
    """복잡 질문 처리 서브그래프 생성

    Args:
        checkpointer: 체크포인터 (interrupt 지원을 위해 필요)

    Returns:
        컴파일된 서브그래프
    """

    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("planning_node", planning_node)
    workflow.add_node("human_approval_node", human_approval_node)
    workflow.add_node("recommend_tool_node", recommend_tool_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("guide_generation_node", guide_generation_node)

    # Entry Point
    workflow.set_entry_point("planning_node")

    # planning -> human_approval -> recommend
    workflow.add_edge("planning_node", "human_approval_node")
    workflow.add_edge("human_approval_node", "recommend_tool_node")

    # ReAct 루프: recommend -> (tool_executor | recommend | guide)
    workflow.add_conditional_edges(
        "recommend_tool_node",
        route_after_recommend,
        {
            "tool_executor": "tool_executor",
            "recommend_tool_node": "recommend_tool_node",
            "guide_generation_node": "guide_generation_node"
        }
    )

    # 루프백: tool_executor -> recommend_tool_node
    workflow.add_edge("tool_executor", "recommend_tool_node")

    # 가이드 생성 후 서브그래프 종료
    workflow.add_edge("guide_generation_node", END)

    # 컴파일 (interrupt_before 포함)
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval_node"]
    )
