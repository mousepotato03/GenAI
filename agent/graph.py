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