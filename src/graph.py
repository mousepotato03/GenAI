"""
Graph - LangGraph 기반 ReAct 에이전트 정의
7개 노드: load_memory, plan, human_review, research, evaluate, guide, synthesize
"""
import json
import uuid
from typing import Annotated, List, Dict, Optional, Any, TypedDict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from src.memory import MemoryManager
from src.tools import hybrid_search, calculate_tools_cost, check_freshness_simple
from src.prompts import (
    PLAN_SYSTEM_PROMPT, PLAN_USER_TEMPLATE,
    GUIDE_TOOL_SYSTEM_PROMPT, GUIDE_TOOL_USER_TEMPLATE,
    GUIDE_FALLBACK_SYSTEM_PROMPT, GUIDE_FALLBACK_USER_TEMPLATE,
    SYNTHESIZE_SYSTEM_PROMPT, SYNTHESIZE_USER_TEMPLATE,
    HUMAN_REVIEW_MESSAGE,
    format_search_results, format_plan_summary, format_user_profile, format_guides
)


# ==================== State 정의 ====================

class SubTask(TypedDict):
    """서브태스크 타입"""
    id: str
    description: str
    category: str
    priority: int
    status: str  # pending, in_progress, completed


class ToolCandidate(TypedDict):
    """도구 후보 타입"""
    name: str
    category: str
    description: str
    pricing: str
    monthly_price: float
    url: str
    score: float


class AgentState(TypedDict):
    """에이전트 상태 정의 (Annotated 사용)"""

    # 메시지 히스토리 (add_messages reducer로 자동 병합)
    messages: Annotated[List[BaseMessage], add_messages]

    # 사용자 정보
    user_id: str
    user_query: str
    user_profile: Optional[Dict]

    # Plan 단계
    plan_analysis: str
    subtasks: List[SubTask]
    plan_approved: bool
    user_feedback: Optional[str]

    # Research 단계
    current_task_idx: int
    search_results: Dict[str, List[ToolCandidate]]  # task_id -> results

    # Evaluate 단계
    evaluation_results: Dict[str, Dict]  # task_id -> {tool_found, top_score, fallback}
    fallback_tasks: List[str]

    # Guide 단계
    guides: Dict[str, str]  # task_id -> guide content

    # Synthesize 단계
    final_response: str

    # 메타데이터
    created_at: str
    error: Optional[str]


# ==================== 전역 설정 ====================

SIMILARITY_THRESHOLD = 0.7
LLM_MODEL = "gpt-4o-mini"

# 메모리 매니저 (싱글톤)
_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """메모리 매니저 싱글톤 반환"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(persist_dir="./db")
    return _memory_manager


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """LLM 인스턴스 반환"""
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)


# ==================== 노드 함수들 ====================

def load_memory_node(state: AgentState) -> Dict:
    """
    [Node 1] 장기 메모리에서 사용자 프로필 로드
    """
    print("[Node] load_memory 실행")

    memory = get_memory_manager()
    user_id = state.get("user_id", "default_user")

    # 사용자 프로필 로드
    profile = memory.load_user_profile(user_id)

    return {
        "user_profile": profile,
        "messages": [
            SystemMessage(content="AI 101 에이전트가 시작되었습니다. 사용자 프로필을 로드했습니다.")
        ]
    }


def plan_node(state: AgentState) -> Dict:
    """
    [Node 2] 사용자 요청을 분석하고 SubTask로 분해
    """
    print("[Node] plan 실행")

    llm = get_llm(temperature=0.5)
    user_query = state.get("user_query", "")
    user_profile = state.get("user_profile")

    # 프롬프트 구성
    user_prompt = PLAN_USER_TEMPLATE.format(
        user_query=user_query,
        user_profile=format_user_profile(user_profile)
    )

    # LLM 호출
    response = llm.invoke([
        SystemMessage(content=PLAN_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    # 응답 파싱
    try:
        response_text = response.content.strip()
        # JSON 블록 추출
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        plan_data = json.loads(response_text)
        analysis = plan_data.get("analysis", "")
        subtasks_raw = plan_data.get("subtasks", [])

        # SubTask 형식으로 변환
        subtasks = []
        for task in subtasks_raw:
            subtasks.append({
                "id": task.get("id", f"task_{len(subtasks)+1}"),
                "description": task.get("description", ""),
                "category": task.get("category", ""),
                "priority": task.get("priority", len(subtasks)+1),
                "status": "pending"
            })

    except Exception as e:
        print(f"Plan 파싱 오류: {e}")
        analysis = "요청 분석 중 오류가 발생했습니다."
        subtasks = [{
            "id": "task_1",
            "description": user_query,
            "category": "text-generation",
            "priority": 1,
            "status": "pending"
        }]

    return {
        "plan_analysis": analysis,
        "subtasks": subtasks,
        "plan_approved": False,
        "messages": [
            AIMessage(content=f"작업 계획을 수립했습니다:\n{format_plan_summary(subtasks)}")
        ]
    }


def human_review_node(state: AgentState) -> Dict:
    """
    [Node 3] Human-in-the-loop: 사용자 승인 대기
    interrupt()를 사용하여 실행 중단
    """
    print("[Node] human_review 실행")

    subtasks = state.get("subtasks", [])
    plan_summary = format_plan_summary(subtasks)

    # 사용자에게 표시할 메시지 구성
    review_message = HUMAN_REVIEW_MESSAGE.format(plan_summary=plan_summary)

    # interrupt로 실행 중단 - 사용자 입력 대기
    user_response = interrupt({
        "message": review_message,
        "plan": subtasks,
        "options": ["approve", "modify", "cancel"]
    })

    # 사용자 응답 처리
    action = user_response.get("action", "approve")
    feedback = user_response.get("feedback", "")

    if action == "cancel":
        return {
            "plan_approved": False,
            "user_feedback": "사용자가 작업을 취소했습니다.",
            "final_response": "작업이 취소되었습니다. 다른 요청이 있으시면 말씀해주세요.",
            "messages": [AIMessage(content="작업이 취소되었습니다.")]
        }

    if action == "modify" and feedback:
        # 수정된 계획 반영 (간단한 처리)
        return {
            "plan_approved": True,
            "user_feedback": feedback,
            "messages": [AIMessage(content=f"수정 사항을 반영하여 진행합니다: {feedback}")]
        }

    return {
        "plan_approved": True,
        "user_feedback": None,
        "messages": [AIMessage(content="계획이 승인되었습니다. 작업을 시작합니다.")]
    }


def research_node(state: AgentState) -> Dict:
    """
    [Node 4] 각 SubTask에 대해 RAG 검색 수행
    """
    print("[Node] research 실행")

    memory = get_memory_manager()
    subtasks = state.get("subtasks", [])

    search_results = {}

    for task in subtasks:
        task_id = task["id"]
        query = task["description"]
        category = task.get("category")

        # 하이브리드 검색 (RAG + Web fallback)
        results, _ = hybrid_search(
            memory_manager=memory,
            query=query,
            k=5,
            threshold=SIMILARITY_THRESHOLD,
            category=category,
            use_web_fallback=True
        )

        search_results[task_id] = results
        print(f"  - {task_id}: {len(results)}개 결과")

    return {
        "search_results": search_results,
        "messages": [AIMessage(content=f"검색 완료: {len(subtasks)}개 작업에 대한 도구 후보를 찾았습니다.")]
    }


def evaluate_node(state: AgentState) -> Dict:
    """
    [Node 5] 검색 결과 평가 및 Threshold 체크
    score >= 0.7: 도구 추천 모드
    score < 0.7: Fallback 모드
    """
    print("[Node] evaluate 실행")

    search_results = state.get("search_results", {})
    evaluation_results = {}
    fallback_tasks = []

    for task_id, results in search_results.items():
        if not results:
            # 검색 결과 없음 → Fallback
            evaluation_results[task_id] = {
                "tool_found": False,
                "top_score": 0,
                "top_tool": None,
                "cost_info": None
            }
            fallback_tasks.append(task_id)
        else:
            top_score = max(r.get("score", 0) for r in results)
            top_tool = max(results, key=lambda x: x.get("score", 0))

            if top_score >= SIMILARITY_THRESHOLD:
                # 도구 추천 모드
                cost_info = calculate_tools_cost(results[:3])
                evaluation_results[task_id] = {
                    "tool_found": True,
                    "top_score": top_score,
                    "top_tool": top_tool,
                    "cost_info": cost_info
                }
            else:
                # Fallback 모드
                evaluation_results[task_id] = {
                    "tool_found": False,
                    "top_score": top_score,
                    "top_tool": None,
                    "cost_info": None
                }
                fallback_tasks.append(task_id)

        print(f"  - {task_id}: score={evaluation_results[task_id]['top_score']:.2f}, "
              f"tool_found={evaluation_results[task_id]['tool_found']}")

    return {
        "evaluation_results": evaluation_results,
        "fallback_tasks": fallback_tasks,
        "messages": [AIMessage(content=f"평가 완료: {len(fallback_tasks)}개 작업이 Fallback 모드입니다.")]
    }


def guide_node(state: AgentState) -> Dict:
    """
    [Node 6] 각 SubTask에 대한 가이드 생성
    - tool_found=True: 도구 추천 가이드
    - tool_found=False: Fallback (일반 조언) 가이드
    """
    print("[Node] guide 실행")

    llm = get_llm(temperature=0.7)
    subtasks = state.get("subtasks", [])
    search_results = state.get("search_results", {})
    evaluation_results = state.get("evaluation_results", {})
    user_profile = state.get("user_profile", {})

    guides = {}

    for task in subtasks:
        task_id = task["id"]
        task_desc = task["description"]
        eval_result = evaluation_results.get(task_id, {})
        results = search_results.get(task_id, [])

        if eval_result.get("tool_found", False):
            # 도구 추천 모드
            user_prompt = GUIDE_TOOL_USER_TEMPLATE.format(
                task_description=task_desc,
                search_results=format_search_results(results[:3]),
                user_preferences=format_user_profile(user_profile)
            )

            response = llm.invoke([
                SystemMessage(content=GUIDE_TOOL_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])
        else:
            # Fallback 모드
            user_prompt = GUIDE_FALLBACK_USER_TEMPLATE.format(
                task_description=task_desc,
                search_results=format_search_results(results) if results else "관련 도구를 찾지 못했습니다."
            )

            response = llm.invoke([
                SystemMessage(content=GUIDE_FALLBACK_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

        guides[task_id] = response.content
        print(f"  - {task_id}: 가이드 생성 완료")

    return {
        "guides": guides,
        "messages": [AIMessage(content=f"가이드 생성 완료: {len(guides)}개 작업")]
    }


def synthesize_node(state: AgentState) -> Dict:
    """
    [Node 7] 모든 가이드를 통합하여 최종 응답 생성 + Reflection 저장
    """
    print("[Node] synthesize 실행")

    llm = get_llm(temperature=0.7)
    memory = get_memory_manager()

    user_query = state.get("user_query", "")
    subtasks = state.get("subtasks", [])
    guides = state.get("guides", {})
    user_id = state.get("user_id", "default_user")
    messages = state.get("messages", [])
    user_profile = state.get("user_profile")

    # 최종 응답 생성
    user_prompt = SYNTHESIZE_USER_TEMPLATE.format(
        original_query=user_query,
        plan=format_plan_summary(subtasks),
        guides=format_guides(guides)
    )

    response = llm.invoke([
        SystemMessage(content=SYNTHESIZE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    final_response = response.content

    # Reflection: 사용자 선호도 추출 및 저장
    try:
        # 대화 내용에서 메시지 추출
        conversation = [
            {"role": "user", "content": user_query}
        ]
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                conversation.append({"role": "assistant", "content": msg.content})

        # 선호도 추출
        new_preferences = memory.extract_preferences(conversation, user_profile)

        # 프로필 저장
        memory.save_user_profile(user_id, new_preferences)
        print(f"  - 사용자 프로필 저장 완료: {user_id}")

    except Exception as e:
        print(f"  - Reflection 오류: {e}")

    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)]
    }


# ==================== 라우팅 함수 ====================

def should_continue_after_review(state: AgentState) -> str:
    """human_review 후 라우팅 결정"""
    if state.get("plan_approved", False):
        return "research"
    else:
        return END


def should_continue_after_evaluate(state: AgentState) -> str:
    """evaluate 후 항상 guide로 진행"""
    return "guide"


# ==================== 그래프 빌드 ====================

def create_agent_graph():
    """LangGraph 에이전트 그래프 생성"""

    # 그래프 정의
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("research", research_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("guide", guide_node)
    workflow.add_node("synthesize", synthesize_node)

    # 엣지 정의
    workflow.set_entry_point("load_memory")

    workflow.add_edge("load_memory", "plan")
    workflow.add_edge("plan", "human_review")

    # human_review 후 조건부 라우팅
    workflow.add_conditional_edges(
        "human_review",
        should_continue_after_review,
        {
            "research": "research",
            END: END
        }
    )

    workflow.add_edge("research", "evaluate")
    workflow.add_edge("evaluate", "guide")
    workflow.add_edge("guide", "synthesize")
    workflow.add_edge("synthesize", END)

    # 체크포인터 (메모리 기반)
    checkpointer = MemorySaver()

    # 그래프 컴파일
    # interrupt_before: human_review 실행 전에 중단하여 plan 결과 확인
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]
    )

    return graph


# ==================== 헬퍼 함수 ====================

def create_initial_state(
    user_query: str,
    user_id: str = "default_user"
) -> AgentState:
    """초기 상태 생성"""
    return {
        "messages": [HumanMessage(content=user_query)],
        "user_id": user_id,
        "user_query": user_query,
        "user_profile": None,
        "plan_analysis": "",
        "subtasks": [],
        "plan_approved": False,
        "user_feedback": None,
        "current_task_idx": 0,
        "search_results": {},
        "evaluation_results": {},
        "fallback_tasks": [],
        "guides": {},
        "final_response": "",
        "created_at": datetime.now().isoformat(),
        "error": None
    }


def run_agent(
    user_query: str,
    user_id: str = "default_user",
    thread_id: Optional[str] = None
):
    """
    에이전트 실행 (동기 방식)

    Args:
        user_query: 사용자 질문
        user_id: 사용자 ID
        thread_id: 스레드 ID (없으면 자동 생성)

    Returns:
        최종 상태
    """
    graph = create_agent_graph()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(user_query, user_id)

    # 그래프 실행 (interrupt까지)
    for event in graph.stream(initial_state, config):
        for node_name, node_output in event.items():
            print(f"[Stream] {node_name}")

    return graph.get_state(config)


async def run_agent_stream(
    user_query: str,
    user_id: str = "default_user",
    thread_id: Optional[str] = None
):
    """
    에이전트 실행 (스트리밍 방식)

    Yields:
        (node_name, output) 튜플
    """
    graph = create_agent_graph()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(user_query, user_id)

    # 비동기 스트리밍
    async for event in graph.astream(initial_state, config):
        for node_name, node_output in event.items():
            yield node_name, node_output


# ==================== 테스트 ====================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # AI 도구 데이터 로드 (최초 1회)
    memory = get_memory_manager()
    json_path = "./data/tools_base.json"
    if os.path.exists(json_path):
        memory.load_tools_from_json(json_path)

    print("\n" + "=" * 50)
    print("AI 101 에이전트 테스트")
    print("=" * 50 + "\n")

    # 테스트 실행
    state = run_agent(
        user_query="유튜브 쇼츠 미스테리 영상을 만들고 싶어",
        user_id="test_user"
    )

    print("\n" + "=" * 50)
    print("최종 상태:")
    print(f"- Plan Approved: {state.values.get('plan_approved')}")
    print(f"- SubTasks: {len(state.values.get('subtasks', []))}")
    print(f"- Fallback Tasks: {state.values.get('fallback_tasks', [])}")
