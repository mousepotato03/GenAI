"""
LangGraph Workflow Integration

모든 노드를 연결하여 전체 워크플로우 그래프를 구성합니다.

워크플로우:
1. Plan → 2. Review → (승인 대기) → 3. Research → 4. Evaluate → 5. Guide → 6. Synthesize → 7. Memory
"""

import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import AgentState, MAX_RETRIES
from src.agents.nodes.planner import PlannerNode
from src.agents.nodes.reviewer import ReviewNode, should_continue
from src.agents.nodes.researcher import ResearchNode
from src.agents.nodes.evaluator import EvaluateNode
from src.agents.nodes.guide_generator import GuideNode
from src.agents.nodes.synthesizer import SynthesizeNode
from src.agents.nodes.memory import MemoryNode, load_user_profile

from src.db.vector_store import VectorStore
from src.tools.search import HybridSearch
from src.tools.evaluator import ReputationEvaluator


def should_retry(state: AgentState) -> str:
    """ReAct 루프: Evaluate 결과를 분석하여 재시도 필요 여부 결정 (Observation 단계)

    Args:
        state: 현재 그래프 상태 (evaluation_results 포함)

    Returns:
        "retry": Plan으로 돌아가서 다른 접근 시도
        "continue": 정상 진행 (Guide로)
    """
    retry_count = state.get("retry_count", 0)
    evaluation_results = state.get("evaluation_results", {})

    # 최대 재시도 횟수 초과
    if retry_count >= MAX_RETRIES:
        print(f"[ReAct] 최대 재시도 횟수 도달 ({retry_count}/{MAX_RETRIES}), 현재 결과로 진행")
        return "continue"

    # 평가 결과 분석
    if not evaluation_results:
        print(f"[ReAct] 평가 결과 없음, 현재 결과로 진행")
        return "continue"

    # 모든 태스크의 평균 점수 계산
    total_score = 0.0
    task_count = 0
    low_score_count = 0

    for task_id, candidates in evaluation_results.items():
        if candidates:
            # 각 태스크의 최고 점수 도구 확인
            top_candidate = candidates[0]
            final_score = top_candidate.get("final_score", 0.0)
            total_score += final_score
            task_count += 1

            # 낮은 점수 체크 (0.6 미만)
            if final_score < 0.6:
                low_score_count += 1

    if task_count == 0:
        print(f"[ReAct] 평가 대상 태스크 없음, 현재 결과로 진행")
        return "continue"

    avg_score = total_score / task_count

    # 재시도 조건:
    # 1. 평균 점수가 0.65 미만이고
    # 2. 낮은 점수 태스크가 전체의 50% 이상일 때
    retry_threshold = 0.65
    low_score_ratio = low_score_count / task_count

    print(f"[ReAct] 평가 결과 - 평균 점수: {avg_score:.2f}, 낮은 점수 비율: {low_score_ratio:.0%}")

    if avg_score < retry_threshold and low_score_ratio >= 0.5:
        print(f"[ReAct] 점수 불충분 → 재시도 {retry_count + 1}회 시작")
        return "retry"
    else:
        print(f"[ReAct] 점수 충분 → 정상 진행")
        return "continue"


def increment_retry(state: AgentState) -> dict:
    """재시도 카운터 증가 노드

    Args:
        state: 현재 상태

    Returns:
        retry_count가 증가된 상태
    """
    retry_count = state.get("retry_count", 0)
    new_count = retry_count + 1
    print(f"[ReAct] 재시도 카운터 증가: {retry_count} → {new_count}")

    return {
        "retry_count": new_count,
        "plan_approved": False  # 재시도 시 승인 초기화
    }


def create_graph():
    """전체 워크플로우 그래프 생성

    Returns:
        컴파일된 LangGraph
    """
    print("[Graph] LangGraph 워크플로우 생성 중...")

    # ===== 의존성 초기화 =====

    # Vector Store
    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    vector_store = VectorStore(persist_directory)

    # Hybrid Search
    google_api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    hybrid_search = HybridSearch(vector_store, google_api_key, search_engine_id)

    # Reputation Evaluator
    reputation_evaluator = ReputationEvaluator()

    # ===== 노드 초기화 =====
    planner = PlannerNode()
    reviewer = ReviewNode()
    researcher = ResearchNode(hybrid_search)
    evaluator = EvaluateNode(reputation_evaluator)
    guide_gen = GuideNode()
    synthesizer = SynthesizeNode()
    memory = MemoryNode(vector_store)

    print("[Graph] 모든 노드 초기화 완료")

    # ===== 그래프 빌드 =====
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("plan", planner)
    workflow.add_node("review", reviewer)
    workflow.add_node("research", researcher)
    workflow.add_node("evaluate", evaluator)
    workflow.add_node("guide", guide_gen)
    workflow.add_node("synthesize", synthesizer)
    workflow.add_node("memory", memory)
    workflow.add_node("increment_retry", increment_retry)  # ReAct 루프용 재시도 카운터

    print("[Graph] 노드 추가 완료 (8개)")

    # ===== 엣지 정의 =====

    # 시작점: Plan
    workflow.set_entry_point("plan")

    # Plan → Review
    workflow.add_edge("plan", "review")

    # Review → 조건부 엣지 (승인 여부)
    workflow.add_conditional_edges(
        "review",
        should_continue,
        {
            "approved": "research",  # 승인됨 → Research로 진행
            "revise": "plan"         # 거절됨 → Plan으로 돌아가서 수정
        }
    )

    # Research → Evaluate
    workflow.add_edge("research", "evaluate")

    # Evaluate → 조건부 엣지 (ReAct 루프: Observation 단계)
    workflow.add_conditional_edges(
        "evaluate",
        should_retry,
        {
            "retry": "increment_retry",  # 재시도: 카운터 증가 후 Plan으로
            "continue": "guide"          # 정상 진행: Guide로
        }
    )

    # increment_retry → Plan (ReAct 루프: 다시 Reasoning 단계로)
    workflow.add_edge("increment_retry", "plan")

    # Guide → Synthesize
    workflow.add_edge("guide", "synthesize")

    # Synthesize → Memory
    workflow.add_edge("synthesize", "memory")

    # Memory → END
    workflow.add_edge("memory", END)

    print("[Graph] 엣지 연결 완료")

    # ===== Checkpointer 설정 (Human-in-the-loop) =====
    checkpointer = MemorySaver()

    # 컴파일
    # interrupt_before=["review"]: Review 노드 전에 실행 중단
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["review"]
    )

    print("[Graph] 그래프 컴파일 완료!")
    print("[Graph] Interrupt 설정: ['review'] 노드 전 중단")

    return graph


def run_graph(user_query: str, session_id: str = "default"):
    """그래프 실행 (테스트용)

    Args:
        user_query: 사용자 질문
        session_id: 세션 ID (체크포인터 식별)

    Returns:
        최종 상태
    """
    print(f"\n{'='*60}")
    print(f"[Graph] 워크플로우 실행 시작")
    print(f"[Graph] 사용자 질문: {user_query}")
    print(f"[Graph] 세션 ID: {session_id}")
    print(f"{'='*60}\n")

    # 그래프 생성
    graph = create_graph()

    # 사용자 프로필 로드
    user_profile = load_user_profile(user_id=session_id)

    # 초기 상태
    initial_state = {
        "user_query": user_query,
        "user_profile": user_profile,
        "plan": [],
        "plan_approved": False,
        "user_feedback": None,
        "research_results": {},
        "evaluation_results": {},
        "guides": {},
        "final_response": "",
        "fallback_tasks": [],
        "error": None,
        "retry_count": 0  # ReAct 루프 초기화
    }

    # Config (세션 ID 지정)
    config = {"configurable": {"thread_id": session_id}}

    # 실행 (interrupt까지)
    print("\n[Graph] Plan 단계 실행 중...\n")
    result = graph.invoke(initial_state, config)

    # 현재 상태 확인
    if not result.get("plan_approved", False):
        print("\n" + "="*60)
        print("[Graph] Review 단계: 계획 승인 대기")
        print("="*60)
        print("\n생성된 계획:")
        for i, task in enumerate(result.get("plan", []), 1):
            print(f"{i}. {task['description']} (카테고리: {task['category']})")

        print("\n계획을 승인하려면:")
        print("1. graph.update_state()로 plan_approved=True 설정")
        print("2. graph.invoke(None, config)로 그래프 재개")
        print("\n또는 FastAPI의 /approve 엔드포인트 사용")

        return result

    # 완료 상태
    print("\n" + "="*60)
    print("[Graph] 워크플로우 완료!")
    print("="*60)
    print("\n최종 응답:\n")
    print(result.get("final_response", "응답을 생성할 수 없습니다."))

    return result


# ===== 테스트 실행 =====
if __name__ == "__main__":
    # 환경변수 로드
    from dotenv import load_dotenv
    load_dotenv()

    # 테스트 쿼리
    test_query = "유튜브 미스테리 쇼츠를 만들고 싶어. 시나리오부터 영상, 더빙까지 전부"

    # 그래프 실행
    result = run_graph(test_query, session_id="test_session_001")

    # Plan 승인 (자동)
    if not result.get("plan_approved"):
        print("\n[테스트] 계획 자동 승인 중...")

        # 그래프 재생성 (동일 세션)
        graph = create_graph()
        config = {"configurable": {"thread_id": "test_session_001"}}

        # 상태 업데이트
        graph.update_state(config, {"plan_approved": True})

        # 재개
        print("\n[Graph] 그래프 재개 중...\n")
        final_result = graph.invoke(None, config)

        print("\n" + "="*60)
        print("[Graph] 워크플로우 완료!")
        print("="*60)
        print("\n최종 응답:\n")
        print(final_result.get("final_response", "응답을 생성할 수 없습니다."))
