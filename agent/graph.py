"""
Agent Graph - LangGraph 메인 그래프 (서브그래프 구조)
"""
import uuid
from typing import Dict, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import llm_router_node, reflection_node
from agent.subgraphs import create_simple_subgraph, create_complex_subgraph
from agent.routing import route_after_llm_router
from agent.hitl import handle_human_feedback


def create_initial_state(user_query: str, user_id: str = "default_user") -> AgentState:
    """초기 상태 생성"""
    return {
        "messages": [HumanMessage(content=user_query)],
        "tool_result": None,
        "is_complex_task": False,
        "sub_tasks": [],
        "tool_recommendations": {},
        "user_feedback": None,
        "retrieved_docs": [],
        "final_guide": None,
        "user_id": user_id,
        "user_query": user_query,
        "user_profile": None,
        "plan_analysis": "",
        "current_task_idx": 0,
        "tool_call_count": 0,
        "simple_tool_count": 0,
        "final_answer": None,
        "created_at": datetime.now().isoformat(),
        "error": None
    }


def create_agent_graph():
    """LangGraph 메인 그래프 생성 (서브그래프 구조)"""

    checkpointer = MemorySaver()

    # 서브그래프 생성
    simple_subgraph = create_simple_subgraph()
    complex_subgraph = create_complex_subgraph(checkpointer=checkpointer)

    # 메인 그래프 정의
    workflow = StateGraph(AgentState)

    # ===== 노드 추가 =====
    workflow.add_node("llm_router", llm_router_node)
    workflow.add_node("simple_subgraph", simple_subgraph)
    workflow.add_node("complex_subgraph", complex_subgraph)
    workflow.add_node("reflection_node", reflection_node)

    # ===== Entry Point =====
    workflow.set_entry_point("llm_router")

    # ===== llm_router 분기 (서브그래프 선택) =====
    workflow.add_conditional_edges(
        "llm_router",
        route_after_llm_router,
        {
            "simple_subgraph": "simple_subgraph",
            "complex_subgraph": "complex_subgraph"
        }
    )

    # ===== 서브그래프 -> reflection =====
    workflow.add_edge("simple_subgraph", "reflection_node")
    workflow.add_edge("complex_subgraph", "reflection_node")

    # ===== 마무리 =====
    workflow.add_edge("reflection_node", END)

    # ===== 컴파일 =====
    graph = workflow.compile(checkpointer=checkpointer)

    return graph


def run_agent(
    user_query: str,
    user_id: str = "default_user",
    thread_id: Optional[str] = None
) -> Dict:
    """
    에이전트 실행 (동기)

    Returns:
        최종 상태
    """
    graph = create_agent_graph()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(user_query, user_id)

    print(f"\n{'='*50}")
    print(f"[Agent] 쿼리: {user_query}")
    print(f"[Agent] Thread: {thread_id}")
    print(f"{'='*50}\n")

    # 그래프 실행 (interrupt까지)
    for event in graph.stream(initial_state, config):
        for node_name, node_output in event.items():
            print(f"[Stream] {node_name} 완료")

    return graph.get_state(config)


async def run_agent_stream(
    user_query: str,
    user_id: str = "default_user",
    thread_id: Optional[str] = None
):
    """
    에이전트 실행 (비동기 스트리밍)

    Yields:
        (node_name, node_output) 튜플
    """
    graph = create_agent_graph()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(user_query, user_id)

    async for event in graph.astream(initial_state, config):
        for node_name, node_output in event.items():
            yield node_name, node_output


def resume_after_approval(
    graph,
    thread_id: str,
    user_response: str
) -> Dict:
    """
    Human-in-the-Loop 승인 후 재개

    Args:
        graph: 컴파일된 그래프
        thread_id: 스레드 ID
        user_response: 사용자 응답

    Returns:
        최종 상태
    """
    config = {"configurable": {"thread_id": thread_id}}

    # 현재 상태 가져오기
    current_state = graph.get_state(config)

    # 피드백 처리
    state_dict = dict(current_state.values)
    updates = handle_human_feedback(state_dict, user_response)

    # 취소인 경우
    if updates.get("error") == "사용자 취소":
        return {"final_guide": updates.get("final_guide", "취소됨")}

    # 상태 업데이트
    graph.update_state(config, updates)

    # 실행 재개
    for event in graph.stream(None, config):
        for node_name, node_output in event.items():
            print(f"[Resume] {node_name} 완료")

    return graph.get_state(config)
