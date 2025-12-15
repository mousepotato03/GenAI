"""
Agent State - 에이전트 상태 정의
"""
from typing import Annotated, List, Dict, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """에이전트 상태 정의"""

    # === 필수 필드 ===
    messages: Annotated[List[BaseMessage], add_messages]  # 단기 메모리
    tool_result: Optional[str]           # Tool Call JSON (ReAct 루프 분기용)
    is_complex_task: bool                # 작업 1: 단순/복잡 판단
    sub_tasks: List[str]                 # 작업 2: 서브태스크 목록 (description만)
    tool_recommendations: Dict[str, str] # 작업 3,4: task_id -> 추천 결과
    user_feedback: Optional[str]         # HITL 피드백
    retrieved_docs: List[Dict]           # RAG 검색 결과
    final_guide: Optional[str]           # 작업 5: 최종 가이드

    # === 유지할 필드 ===
    user_id: str
    user_query: str
    user_profile: Optional[Dict]

    # === 확장 필드 ===
    plan_analysis: str                   # 계획 분석 결과
    current_task_idx: int                # 현재 처리 중인 태스크 인덱스
    tool_call_count: int                 # ReAct 루프 카운터 (무한 방지)
    task_completed: bool                 # 태스크 완료 플래그 (executor → recommend 전달용)

    # === 단순 질문 ReAct 필드 ===
    simple_tool_count: int               # 단순 질문용 도구 호출 카운터
    final_answer: Optional[str]          # simple_llm_node의 최종 답변

    # === 메타데이터 ===
    error: Optional[str]
