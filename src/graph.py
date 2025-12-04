"""
Graph - LangGraph 기반 ReAct 에이전트 정의

6개 노드:
- llm_router: 작업 유형 분류 (단순 Q&A vs 복잡 작업)
- planning_node: 서브태스크 분해
- recommend_tool_node: ReAct Agent - 도구 추천
- tool_executor: 도구 실행
- guide_generation_node: 최종 가이드 생성
- reflection_node: 메모리 저장
"""
import json
import uuid
from typing import Annotated, List, Dict, Optional, Any, TypedDict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from src.memory import MemoryManager
from src.tools import (
    get_all_tools, execute_tool,
    retrieve_docs, read_memory, write_memory, google_search_tool,
    calculate_subscription_cost, check_tool_freshness
)
from src.prompts import (
    # 기존 프롬프트
    PLAN_SYSTEM_PROMPT, PLAN_USER_TEMPLATE,
    SYNTHESIZE_SYSTEM_PROMPT, SYNTHESIZE_USER_TEMPLATE,
    HUMAN_REVIEW_MESSAGE,
    INTENT_ANALYSIS_SYSTEM_PROMPT, INTENT_ANALYSIS_USER_TEMPLATE,
    MODIFY_PLAN_SYSTEM_PROMPT, MODIFY_PLAN_USER_TEMPLATE,
    format_search_results, format_plan_summary, format_user_profile, format_guides,
    # 새 프롬프트
    LLM_ROUTER_SYSTEM_PROMPT, LLM_ROUTER_USER_TEMPLATE,
    RECOMMEND_TOOL_SYSTEM_PROMPT, RECOMMEND_TOOL_USER_TEMPLATE,
    GUIDE_GENERATION_SYSTEM_PROMPT, GUIDE_GENERATION_USER_TEMPLATE,
    GUIDE_SIMPLE_QA_SYSTEM_PROMPT, GUIDE_SIMPLE_QA_USER_TEMPLATE,
    MEMORY_EXTRACTOR_SYSTEM_PROMPT, MEMORY_EXTRACTOR_USER_TEMPLATE
)


# ==================== State 정의 (guide.md 스펙) ====================

class AgentState(TypedDict):
    """에이전트 상태 정의 (guide.md 스펙)"""

    # === guide.md 필수 필드 ===
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

    # === 메타데이터 ===
    created_at: str
    error: Optional[str]


# ==================== 전역 설정 ====================

SIMILARITY_THRESHOLD = 0.7
LLM_MODEL = "gpt-4o-mini"
MAX_TOOL_CALLS_PER_TASK = 3  # ReAct 루프 무한 방지

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


# ==================== 헬퍼 함수 ====================

def analyze_user_intent(user_text: str, plan_summary: str) -> dict:
    """사용자의 자연어 응답에서 의도를 분석"""
    llm = get_llm(temperature=0.1)

    user_prompt = INTENT_ANALYSIS_USER_TEMPLATE.format(
        plan_summary=plan_summary,
        user_response=user_text
    )

    response = llm.invoke([
        SystemMessage(content=INTENT_ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    try:
        response_text = response.content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        result = json.loads(response_text)
        return {
            "action": result.get("intent", "approve"),
            "feedback": result.get("feedback", "")
        }
    except Exception as e:
        print(f"의도 분석 파싱 오류: {e}")
        return {"action": "approve", "feedback": ""}


def modify_subtasks(current_subtasks: List[str], feedback: str) -> List[str]:
    """사용자 피드백을 반영하여 서브태스크 수정"""
    llm = get_llm(temperature=0.3)

    current_plan = "\n".join([f"- {task}" for task in current_subtasks])

    user_prompt = MODIFY_PLAN_USER_TEMPLATE.format(
        current_plan=current_plan,
        feedback=feedback
    )

    response = llm.invoke([
        SystemMessage(content=MODIFY_PLAN_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    try:
        response_text = response.content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        modified_tasks = json.loads(response_text)

        if isinstance(modified_tasks, list):
            # [{id, description, ...}] 형식이면 description만 추출
            if modified_tasks and isinstance(modified_tasks[0], dict):
                return [t.get("description", str(t)) for t in modified_tasks]
            return modified_tasks

        return current_subtasks
    except Exception as e:
        print(f"계획 수정 파싱 오류: {e}")
        return current_subtasks


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
        "created_at": datetime.now().isoformat(),
        "error": None
    }


# ==================== 노드 구현 ====================

def llm_router_node(state: AgentState) -> Dict:
    """
    [작업 1] 질문 유형 분류 - Entry Point

    사용자 질문이 단순 Q&A인지 복잡한 작업인지 판단합니다.
    """
    print("[Node] llm_router 실행")

    llm = get_llm(temperature=0.1)
    user_query = state.get("user_query", "")

    # 사용자 프로필 로드
    memory = get_memory_manager()
    user_id = state.get("user_id", "default_user")
    user_profile = memory.load_user_profile(user_id)

    user_prompt = LLM_ROUTER_USER_TEMPLATE.format(
        user_query=user_query,
        user_profile=format_user_profile(user_profile)
    )

    response = llm.invoke([
        SystemMessage(content=LLM_ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    try:
        response_text = response.content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        result = json.loads(response_text)
        is_complex = result.get("is_complex", False)
        reason = result.get("reason", "")
        print(f"  - 분류 결과: {'복잡한 작업' if is_complex else '단순 Q&A'}")
        print(f"  - 이유: {reason}")
    except Exception as e:
        print(f"  - 분류 파싱 오류: {e}, 기본값: 복잡한 작업")
        is_complex = True

    return {
        "is_complex_task": is_complex,
        "user_profile": user_profile,
        "messages": [AIMessage(content=f"[시스템] 질문 유형: {'복잡한 작업' if is_complex else '단순 Q&A'}")]
    }


def planning_node(state: AgentState) -> Dict:
    """
    [작업 2] 서브태스크 분해

    복잡한 작업을 2-5개의 서브태스크로 분해합니다.
    """
    print("[Node] planning 실행")

    llm = get_llm(temperature=0.5)
    user_query = state.get("user_query", "")
    user_profile = state.get("user_profile")

    user_prompt = PLAN_USER_TEMPLATE.format(
        user_query=user_query,
        user_profile=format_user_profile(user_profile)
    )

    response = llm.invoke([
        SystemMessage(content=PLAN_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    try:
        response_text = response.content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        plan_data = json.loads(response_text)
        analysis = plan_data.get("analysis", "")
        subtasks_data = plan_data.get("subtasks", [])

        # sub_tasks: description 리스트로 변환
        sub_tasks = [t.get("description", "") for t in subtasks_data]

        print(f"  - 분석: {analysis}")
        print(f"  - 서브태스크 {len(sub_tasks)}개 생성")
        for i, task in enumerate(sub_tasks, 1):
            print(f"    {i}. {task}")

    except Exception as e:
        print(f"  - 계획 파싱 오류: {e}")
        analysis = "계획 수립 중 오류 발생"
        sub_tasks = [user_query]  # 원본 쿼리를 단일 태스크로

    # 계획 요약 메시지 생성
    plan_summary = "\n".join([f"{i+1}. {task}" for i, task in enumerate(sub_tasks)])
    plan_message = f"""## 작업 계획

**분석**: {analysis}

**수립된 계획**:
{plan_summary}

이 계획대로 진행할까요? (승인/수정/취소)"""

    return {
        "plan_analysis": analysis,
        "sub_tasks": sub_tasks,
        "tool_recommendations": {},
        "current_task_idx": 0,
        "tool_call_count": 0,
        "messages": [AIMessage(content=plan_message)]
    }


def recommend_tool_node(state: AgentState) -> Dict:
    """
    [작업 3, 4] ReAct Agent - 도구 추천

    각 서브태스크에 대해 도구를 호출하여 최적의 AI 도구를 추천합니다.
    """
    print("[Node] recommend_tool 실행")

    sub_tasks = state.get("sub_tasks", [])
    current_idx = state.get("current_task_idx", 0)
    tool_call_count = state.get("tool_call_count", 0)
    tool_recommendations = state.get("tool_recommendations", {})
    retrieved_docs = state.get("retrieved_docs", [])
    user_profile = state.get("user_profile")

    # 모든 태스크 완료 체크
    if current_idx >= len(sub_tasks):
        print("  - 모든 서브태스크 처리 완료")
        return {"tool_result": None}

    current_task = sub_tasks[current_idx]
    print(f"  - 현재 태스크: {current_task} ({current_idx + 1}/{len(sub_tasks)})")

    # 무한 루프 방지
    if tool_call_count >= MAX_TOOL_CALLS_PER_TASK:
        print(f"  - 최대 도구 호출 횟수 도달 ({MAX_TOOL_CALLS_PER_TASK}), 다음 태스크로 이동")
        task_id = f"task_{current_idx + 1}"
        tool_recommendations[task_id] = "도구 검색 결과를 바탕으로 직접 확인이 필요합니다."
        return {
            "tool_result": None,
            "tool_recommendations": tool_recommendations,
            "current_task_idx": current_idx + 1,
            "tool_call_count": 0
        }

    # LLM에 도구 바인딩
    tools = get_all_tools()
    llm_with_tools = get_llm(temperature=0.3).bind_tools(tools)

    # 이전 검색 결과 컨텍스트
    previous_results = ""
    if retrieved_docs:
        recent_docs = retrieved_docs[-5:]  # 최근 5개
        previous_results = json.dumps(recent_docs, ensure_ascii=False, indent=2)

    user_prompt = RECOMMEND_TOOL_USER_TEMPLATE.format(
        current_task=current_task,
        previous_results=previous_results if previous_results else "없음",
        user_profile=format_user_profile(user_profile)
    )

    response = llm_with_tools.invoke([
        SystemMessage(content=RECOMMEND_TOOL_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    # Tool Call 확인
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"  - Tool Call: {tool_call['name']}")
        print(f"  - Args: {tool_call['args']}")

        return {
            "tool_result": json.dumps({
                "id": tool_call.get("id", str(uuid.uuid4())),
                "name": tool_call["name"],
                "arguments": tool_call["args"]
            }),
            "tool_call_count": tool_call_count + 1,
            "messages": [response]
        }
    else:
        # Tool Call 없이 추천 완료 -> 다음 태스크로
        print(f"  - 추천 완료, 다음 태스크로 이동")
        task_id = f"task_{current_idx + 1}"
        tool_recommendations[task_id] = response.content

        return {
            "tool_result": None,
            "tool_recommendations": tool_recommendations,
            "current_task_idx": current_idx + 1,
            "tool_call_count": 0,
            "messages": [response]
        }


def tool_executor_node(state: AgentState) -> Dict:
    """
    도구 실행 노드

    recommend_tool_node가 요청한 Tool Call을 실제로 실행합니다.
    """
    print("[Node] tool_executor 실행")

    tool_result = state.get("tool_result")
    retrieved_docs = state.get("retrieved_docs", [])

    if not tool_result:
        print("  - tool_result 없음, 스킵")
        return {}

    try:
        tool_call = json.loads(tool_result)
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        tool_id = tool_call.get("id", "")

        print(f"  - 실행 도구: {tool_name}")
        print(f"  - 인자: {tool_args}")

        # 도구 실행
        result = execute_tool(tool_name, tool_args)
        observation = str(result)

        print(f"  - 결과 길이: {len(observation)} chars")

        # retrieved_docs 업데이트 (검색 결과인 경우)
        if tool_name in ["retrieve_docs", "google_search_tool"]:
            try:
                result_data = json.loads(observation)
                docs = result_data.get("results", [])
                if isinstance(docs, list):
                    retrieved_docs.extend(docs)
                    print(f"  - retrieved_docs에 {len(docs)}개 추가")
            except:
                pass

        # ToolMessage로 Observation 반환
        return {
            "retrieved_docs": retrieved_docs,
            "messages": [ToolMessage(content=observation, tool_call_id=tool_id)],
            "tool_result": None  # 다음 루프를 위해 초기화
        }

    except Exception as e:
        print(f"  - 도구 실행 오류: {e}")
        return {
            "messages": [ToolMessage(content=f"도구 실행 오류: {str(e)}", tool_call_id="error")],
            "tool_result": None,
            "error": str(e)
        }


def guide_generation_node(state: AgentState) -> Dict:
    """
    [작업 5] 최종 가이드 생성

    수집된 정보를 바탕으로 최종 워크플로우 가이드를 생성합니다.
    """
    print("[Node] guide_generation 실행")

    llm = get_llm(temperature=0.7)
    is_complex = state.get("is_complex_task", False)
    user_query = state.get("user_query", "")
    user_profile = state.get("user_profile")
    retrieved_docs = state.get("retrieved_docs", [])

    if not is_complex:
        # 단순 Q&A: 간단한 답변 생성
        print("  - 단순 Q&A 모드")

        # 검색 수행 (단순 Q&A에도 컨텍스트 필요)
        if not retrieved_docs:
            from src.tools import hybrid_search
            memory = get_memory_manager()
            results, _ = hybrid_search(
                memory_manager=memory,
                query=user_query,
                k=3,
                threshold=0.5,
                use_web_fallback=False
            )
            retrieved_docs = results

        user_prompt = GUIDE_SIMPLE_QA_USER_TEMPLATE.format(
            user_query=user_query,
            retrieved_docs=json.dumps(retrieved_docs[:5], ensure_ascii=False, indent=2) if retrieved_docs else "검색 결과 없음",
            user_profile=format_user_profile(user_profile)
        )

        response = llm.invoke([
            SystemMessage(content=GUIDE_SIMPLE_QA_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        return {
            "final_guide": response.content,
            "retrieved_docs": retrieved_docs,
            "messages": [response]
        }

    # 복잡한 작업: 상세 가이드 생성
    print("  - 복잡한 작업 모드")

    sub_tasks = state.get("sub_tasks", [])
    tool_recommendations = state.get("tool_recommendations", {})

    # 서브태스크 포맷팅
    sub_tasks_formatted = "\n".join([f"{i+1}. {task}" for i, task in enumerate(sub_tasks)])

    # 추천 도구 포맷팅
    recommendations_formatted = ""
    for task_id, recommendation in tool_recommendations.items():
        recommendations_formatted += f"\n### {task_id}\n{recommendation}\n"

    user_prompt = GUIDE_GENERATION_USER_TEMPLATE.format(
        user_query=user_query,
        sub_tasks=sub_tasks_formatted,
        tool_recommendations=recommendations_formatted if recommendations_formatted else "추천 정보 없음",
        retrieved_docs=json.dumps(retrieved_docs[:10], ensure_ascii=False, indent=2) if retrieved_docs else "검색 결과 없음",
        user_profile=format_user_profile(user_profile)
    )

    response = llm.invoke([
        SystemMessage(content=GUIDE_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    return {
        "final_guide": response.content,
        "messages": [response]
    }


def reflection_node(state: AgentState) -> Dict:
    """
    [작업 6] Reflection - 메모리 저장

    대화 내용을 분석하여 사용자 선호도를 추출하고 장기 메모리에 저장합니다.
    """
    print("[Node] reflection 실행")

    memory = get_memory_manager()
    user_id = state.get("user_id", "default_user")
    user_query = state.get("user_query", "")
    user_profile = state.get("user_profile")
    final_guide = state.get("final_guide", "")

    # 대화 내용 구성
    conversation = f"사용자: {user_query}\n\n에이전트: {final_guide}"

    llm = get_llm(temperature=0.3)

    user_prompt = MEMORY_EXTRACTOR_USER_TEMPLATE.format(
        conversation=conversation,
        existing_profile=json.dumps(user_profile, ensure_ascii=False, indent=2) if user_profile else "없음"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=MEMORY_EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        response_text = response.content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        new_preferences = json.loads(response_text)

        # 기존 프로필과 병합
        if user_profile:
            merged_profile = user_profile.copy()
            for key, value in new_preferences.items():
                if isinstance(value, list) and isinstance(merged_profile.get(key), list):
                    # 리스트는 합집합
                    merged_profile[key] = list(set(merged_profile[key] + value))
                elif value:  # 빈 값이 아니면 업데이트
                    merged_profile[key] = value
            new_preferences = merged_profile

        # 저장
        success = memory.save_user_profile(user_id, new_preferences)
        if success:
            print(f"  - 사용자 프로필 저장 완료: {user_id}")
        else:
            print(f"  - 사용자 프로필 저장 실패")

    except Exception as e:
        print(f"  - Reflection 오류: {e}")

    return {}


# ==================== 라우팅 함수 ====================

def route_after_llm_router(state: AgentState) -> str:
    """llm_router 후 라우팅"""
    if state.get("is_complex_task", False):
        return "planning_node"
    else:
        return "guide_generation_node"


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


# ==================== Human-in-the-Loop ====================

def handle_human_feedback(state: AgentState, user_response: str) -> Dict:
    """
    사용자 피드백 처리 (interrupt 후 호출)

    Args:
        state: 현재 상태
        user_response: 사용자 응답 (자연어)

    Returns:
        업데이트할 상태 딕셔너리
    """
    sub_tasks = state.get("sub_tasks", [])
    plan_summary = "\n".join([f"{i+1}. {task}" for i, task in enumerate(sub_tasks)])

    # 의도 분석
    intent_result = analyze_user_intent(user_response, plan_summary)
    action = intent_result["action"]
    feedback = intent_result["feedback"]

    print(f"[Human Feedback] action={action}, feedback={feedback}")

    if action == "cancel":
        return {
            "final_guide": "작업이 취소되었습니다. 다른 질문이 있으시면 말씀해주세요.",
            "error": "사용자 취소"
        }
    elif action == "modify":
        # 계획 수정
        modified_tasks = modify_subtasks(sub_tasks, feedback)
        return {
            "sub_tasks": modified_tasks,
            "user_feedback": feedback,
            "current_task_idx": 0,
            "tool_recommendations": {},
            "messages": [AIMessage(content=f"계획이 수정되었습니다:\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(modified_tasks)]))]
        }
    else:
        # 승인
        return {
            "user_feedback": None,
            "messages": [AIMessage(content="계획이 승인되었습니다. 도구 추천을 시작합니다.")]
        }


# ==================== 그래프 빌드 ====================

def create_agent_graph():
    """LangGraph 에이전트 그래프 생성 (guide.md 스펙)"""

    # 그래프 정의
    workflow = StateGraph(AgentState)

    # ===== 노드 추가 =====
    workflow.add_node("llm_router", llm_router_node)
    workflow.add_node("planning_node", planning_node)
    workflow.add_node("recommend_tool_node", recommend_tool_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("guide_generation_node", guide_generation_node)
    workflow.add_node("reflection_node", reflection_node)

    # ===== Entry Point =====
    workflow.set_entry_point("llm_router")

    # ===== llm_router 분기 =====
    workflow.add_conditional_edges(
        "llm_router",
        route_after_llm_router,
        {
            "planning_node": "planning_node",
            "guide_generation_node": "guide_generation_node"
        }
    )

    # ===== planning -> recommend (interrupt 전) =====
    workflow.add_edge("planning_node", "recommend_tool_node")

    # ===== ReAct 루프 =====
    workflow.add_conditional_edges(
        "recommend_tool_node",
        route_after_recommend,
        {
            "tool_executor": "tool_executor",
            "recommend_tool_node": "recommend_tool_node",
            "guide_generation_node": "guide_generation_node"
        }
    )

    # tool_executor -> recommend_tool_node (Loop back)
    workflow.add_edge("tool_executor", "recommend_tool_node")

    # ===== 마무리 =====
    workflow.add_edge("guide_generation_node", "reflection_node")
    workflow.add_edge("reflection_node", END)

    # ===== 컴파일 =====
    checkpointer = MemorySaver()

    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["recommend_tool_node"]  # planning 후 interrupt
    )

    return graph


# ==================== 실행 함수 ====================

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


# ==================== 테스트 ====================

if __name__ == "__main__":
    # 단순 Q&A 테스트
    print("\n" + "="*60)
    print("테스트 1: 단순 Q&A")
    print("="*60)

    result = run_agent("ChatGPT 가격이 얼마야?")
    print(f"\n최종 가이드:\n{result.values.get('final_guide', 'N/A')}")

    # 복잡한 작업 테스트
    print("\n" + "="*60)
    print("테스트 2: 복잡한 작업 (interrupt됨)")
    print("="*60)

    result = run_agent("유튜브 쇼츠 미스테리 영상을 만들고 싶어")
    print(f"\n계획:\n{result.values.get('sub_tasks', [])}")
    print(f"\ninterrupt 상태 (승인 대기 중)")
