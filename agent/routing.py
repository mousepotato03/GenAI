"""
Routing Functions - 메인 그래프 라우팅 로직
"""
from agent.state import AgentState


def route_after_llm_router(state: AgentState) -> str:
    """llm_router 후 서브그래프 선택"""
    if state.get("is_complex_task", False):
        return "complex_subgraph"
    else:
        return "simple_subgraph"
