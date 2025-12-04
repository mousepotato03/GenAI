"""
API Routes - FastAPI 엔드포인트 정의
"""
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import gradio as gr

from app.api.schemas import ChatRequest, ApproveRequest, ChatResponse, HealthResponse
from app.ui.gradio_app import create_gradio_ui
from agent.graph import create_agent_graph, create_initial_state
from agent.hitl import handle_human_feedback
from core.memory import get_memory_manager
from core.config import TOOLS_JSON_PATH, DATA_PATH


# 활성 세션 저장 (thread_id -> graph_state)
active_sessions = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 로직"""
    # 시작 시: AI 도구 데이터 로드
    print("AI 101 에이전트 시작...")
    memory = get_memory_manager()

    # 1. JSON 도구 데이터 로드
    if os.path.exists(TOOLS_JSON_PATH):
        count = memory.load_tools_from_json(TOOLS_JSON_PATH)
        print(f"AI 도구 데이터 로드 완료: {count}개")
    else:
        print(f"경고: {TOOLS_JSON_PATH} 파일을 찾을 수 없습니다.")

    # 2. PDF 지식베이스 로드
    pdf_count = memory.load_pdfs_from_directory(DATA_PATH)
    print(f"PDF 지식베이스 로드 완료: {pdf_count}개 청크")

    yield

    # 종료 시
    print("AI 101 에이전트 종료...")


def create_app() -> FastAPI:
    """FastAPI 앱 생성"""
    app = FastAPI(
        title="AI 101",
        description="LangGraph 기반 지능형 AI 도구 추천 에이전트",
        version="1.0.0",
        lifespan=lifespan
    )

    # 라우트 등록
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """헬스 체크"""
        memory = get_memory_manager()
        return HealthResponse(
            status="healthy",
            tools_count=memory.get_tools_count(),
            profiles_count=memory.get_profiles_count()
        )

    @app.post("/chat/start", response_model=ChatResponse)
    async def start_chat(request: ChatRequest):
        """채팅 시작 - Plan 단계까지 실행 후 승인 대기"""
        thread_id = request.thread_id or str(uuid.uuid4())

        try:
            graph = create_agent_graph()
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = create_initial_state(request.query, request.user_id)

            # Plan 단계까지 실행 (recommend_tool_node 전에 interrupt)
            for event in graph.stream(initial_state, config):
                for node_name, node_output in event.items():
                    print(f"[{thread_id}] Node: {node_name}")

            # 현재 상태 조회
            state = graph.get_state(config)
            is_complex = state.values.get("is_complex_task", False)
            sub_tasks = state.values.get("sub_tasks", [])
            plan_analysis = state.values.get("plan_analysis", "")
            final_guide = state.values.get("final_guide")

            # 단순 Q&A인 경우 바로 완료
            if not is_complex and final_guide:
                return ChatResponse(
                    thread_id=thread_id,
                    status="completed",
                    message="답변이 완료되었습니다.",
                    is_complex=False,
                    final_guide=final_guide
                )

            # 복잡한 작업: 세션 저장 및 승인 대기
            active_sessions[thread_id] = {
                "graph": graph,
                "config": config,
                "state": state
            }

            return ChatResponse(
                thread_id=thread_id,
                status="pending_approval",
                message=f"작업 계획을 수립했습니다.\n\n분석: {plan_analysis}\n\n승인하시겠습니까?",
                is_complex=True,
                plan=[{"id": f"task_{i+1}", "description": task} for i, task in enumerate(sub_tasks)]
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/chat/approve", response_model=ChatResponse)
    async def approve_plan(request: ApproveRequest):
        """계획 승인 후 나머지 단계 실행"""
        thread_id = request.thread_id

        if thread_id not in active_sessions:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        session = active_sessions[thread_id]
        graph = session["graph"]
        config = session["config"]

        try:
            # 사용자 응답 텍스트 구성
            if request.action == "cancel":
                user_response = "취소해"
            elif request.action == "modify" and request.feedback:
                user_response = request.feedback
            else:
                user_response = "승인"

            # handle_human_feedback로 상태 업데이트
            current_state = graph.get_state(config)
            state_dict = dict(current_state.values)
            updates = handle_human_feedback(state_dict, user_response)

            # 취소인 경우
            if updates.get("error") == "사용자 취소":
                if thread_id in active_sessions:
                    del active_sessions[thread_id]
                return ChatResponse(
                    thread_id=thread_id,
                    status="cancelled",
                    message="작업이 취소되었습니다.",
                    is_complex=True,
                    final_guide=None
                )

            # 상태 업데이트 후 실행 재개
            graph.update_state(config, updates)

            for event in graph.stream(None, config):
                for node_name, node_output in event.items():
                    print(f"[{thread_id}] Node: {node_name}")

            # 최종 상태 조회
            state = graph.get_state(config)
            final_guide = state.values.get("final_guide", "")

            # 세션 정리
            if thread_id in active_sessions:
                del active_sessions[thread_id]

            return ChatResponse(
                thread_id=thread_id,
                status="completed",
                message="작업이 완료되었습니다.",
                is_complex=True,
                final_guide=final_guide
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Gradio 앱 생성 및 마운트
    gradio_app = create_gradio_ui(active_sessions)
    app = gr.mount_gradio_app(app, gradio_app, path="/")

    return app
