"""
Simple Subgraph - 단순 질문 처리 서브그래프
"""
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import simple_llm_node, simple_tool_executor


def route_after_simple_llm(state: AgentState) -> str:
    """simple_llm_node 후 라우팅 (단순 질문 ReAct 루프)"""
    tool_result = state.get("tool_result")

    if tool_result is not None:
        return "simple_executor"
    else:
        # 최종 답변 완료 -> 서브그래프 종료
        return END


def create_simple_subgraph():
    """단순 질문 처리 서브그래프 생성"""

    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("simple_llm_node", simple_llm_node)
    workflow.add_node("simple_executor", simple_tool_executor)

    # Entry Point
    workflow.set_entry_point("simple_llm_node")

    # 조건부 엣지: simple_llm_node 후 분기
    workflow.add_conditional_edges(
        "simple_llm_node",
        route_after_simple_llm,
        {
            "simple_executor": "simple_executor",
            END: END
        }
    )

    # 루프백: simple_executor -> simple_llm_node
    workflow.add_edge("simple_executor", "simple_llm_node")

    return workflow.compile()
