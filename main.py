"""
AI 101 - 지능형 AI 도구 추천 에이전트
FastAPI + Gradio UI 통합 엔트리포인트
"""
import os
import json
import uuid
import asyncio
from typing import Optional, Generator, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import gradio as gr
from dotenv import load_dotenv

from src.graph import (
    create_agent_graph,
    create_initial_state,
    get_memory_manager,
    handle_human_feedback,
    resume_after_approval
)
from src.prompts import format_plan_summary

# 환경변수 로드
load_dotenv()


# ==================== FastAPI 설정 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 로직"""
    # 시작 시: AI 도구 데이터 로드
    print("AI 101 에이전트 시작...")
    memory = get_memory_manager()

    # 1. JSON 도구 데이터 로드
    json_path = "./data/ai_tools_2025.json"
    if os.path.exists(json_path):
        count = memory.load_tools_from_json(json_path)
        print(f"AI 도구 데이터 로드 완료: {count}개")
    else:
        print(f"경고: {json_path} 파일을 찾을 수 없습니다.")

    # 2. PDF 지식베이스 로드
    pdf_dir = "./data"
    pdf_count = memory.load_pdfs_from_directory(pdf_dir)
    print(f"PDF 지식베이스 로드 완료: {pdf_count}개 청크")

    yield

    # 종료 시
    print("AI 101 에이전트 종료...")


app = FastAPI(
    title="AI 101",
    description="LangGraph 기반 지능형 AI 도구 추천 에이전트",
    version="1.0.0",
    lifespan=lifespan
)


# ==================== API 모델 ====================

class ChatRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    thread_id: Optional[str] = None


class ApproveRequest(BaseModel):
    thread_id: str
    action: str = "approve"  # approve, modify, cancel
    feedback: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    status: str
    message: str
    is_complex: bool = False
    plan: Optional[list] = None
    final_guide: Optional[str] = None


# ==================== 세션 관리 ====================

# 활성 세션 저장 (thread_id -> graph_state)
active_sessions = {}


# ==================== API 엔드포인트 ====================

@app.get("/health")
async def health_check():
    """헬스 체크"""
    memory = get_memory_manager()
    return {
        "status": "healthy",
        "tools_count": memory.get_tools_count(),
        "profiles_count": memory.get_profiles_count()
    }


@app.post("/chat/start", response_model=ChatResponse)
async def start_chat(request: ChatRequest):
    """
    채팅 시작 - Plan 단계까지 실행 후 승인 대기
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        graph = create_agent_graph()
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = create_initial_state(request.query, request.user_id)

        # Plan 단계까지 실행 (recommend_tool_node 전에 interrupt)
        final_state = None
        for event in graph.stream(initial_state, config):
            for node_name, node_output in event.items():
                print(f"[{thread_id}] Node: {node_name}")
            final_state = event

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
    """
    계획 승인 후 나머지 단계 실행
    """
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


# ==================== Gradio UI ====================

def create_gradio_ui():
    """Gradio 채팅 인터페이스 생성 - 완전 자연어 방식"""

    def process_message(message: str, history: list, user_id: str, thread_id: str):
        """
        통합 메시지 처리
        - 세션 없음: 새 대화 시작 (Plan 생성)
        - 세션 있음: 계획에 대한 자연어 응답 처리
        """
        if not message.strip():
            return history, thread_id, "메시지를 입력해주세요."

        history = history or []

        # 활성 세션이 있으면 → 계획에 대한 응답으로 처리
        if thread_id and thread_id in active_sessions:
            return continue_with_response(message, history, thread_id)

        # 새 대화 시작
        return start_new_conversation(message, history, user_id)

    def start_new_conversation(message: str, history: list, user_id: str):
        """새 대화 시작 - Plan 생성 또는 단순 Q&A 처리"""
        thread_id = str(uuid.uuid4())

        try:
            graph = create_agent_graph()
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = create_initial_state(message, user_id or "gradio_user")

            # 그래프 실행 (interrupt까지 또는 완료까지)
            for event in graph.stream(initial_state, config):
                pass

            state = graph.get_state(config)
            is_complex = state.values.get("is_complex_task", False)
            sub_tasks = state.values.get("sub_tasks", [])
            plan_analysis = state.values.get("plan_analysis", "")
            final_guide = state.values.get("final_guide")

            # 히스토리 업데이트
            history.append({"role": "user", "content": message})

            # 단순 Q&A: 바로 응답
            if not is_complex and final_guide:
                history.append({"role": "assistant", "content": final_guide})
                return history, None, "완료!"

            # 복잡한 작업: 세션 저장 및 승인 대기
            active_sessions[thread_id] = {
                "graph": graph,
                "config": config,
                "state": state
            }

            # 계획 메시지 생성 (자연어 안내)
            plan_text = f"**작업 분석**\n{plan_analysis}\n\n**수립된 계획:**\n"
            for i, task in enumerate(sub_tasks, 1):
                plan_text += f"- **task_{i}**: {task}\n"
            plan_text += "\n이대로 진행할까요? (예: '좋아 진행해', '취소해', '2번은 빼줘' 등으로 응답)"

            history.append({"role": "assistant", "content": plan_text})

            return history, thread_id, "계획 검토 중 - 자연어로 응답해주세요"

        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})
            return history, None, f"오류: {str(e)}"

    def continue_with_response(message: str, history: list, thread_id: str):
        """계획에 대한 자연어 응답 처리"""
        if thread_id not in active_sessions:
            return history, None, "세션이 만료되었습니다. 새로 시작해주세요."

        session = active_sessions[thread_id]
        graph = session["graph"]
        config = session["config"]

        try:
            # 히스토리에 사용자 메시지 추가
            history.append({"role": "user", "content": message})

            # handle_human_feedback로 상태 업데이트
            current_state = graph.get_state(config)
            state_dict = dict(current_state.values)
            updates = handle_human_feedback(state_dict, message)

            # 취소인 경우
            if updates.get("error") == "사용자 취소":
                if thread_id in active_sessions:
                    del active_sessions[thread_id]
                history.append({"role": "assistant", "content": "작업이 취소되었습니다. 새로운 요청이 있으시면 말씀해주세요."})
                return history, None, "취소됨"

            # 상태 업데이트 후 실행 재개
            graph.update_state(config, updates)

            for event in graph.stream(None, config):
                pass

            state = graph.get_state(config)
            final_guide = state.values.get("final_guide", "")

            # 세션 정리
            if thread_id in active_sessions:
                del active_sessions[thread_id]

            # 완료된 경우
            history.append({"role": "assistant", "content": final_guide})
            return history, None, "완료!"

        except Exception as e:
            history.append({"role": "assistant", "content": f"오류: {str(e)}"})
            return history, None, f"오류: {str(e)}"

    # UI 구성 (버튼 제거, 입력창만)
    with gr.Blocks(
        title="AI 101 - AI 도구 추천 에이전트"
    ) as demo:

        # 상태 변수
        current_thread_id = gr.State(None)

        gr.Markdown("""
        # AI 101 - 지능형 AI 도구 추천 에이전트

        AI 도구를 활용한 작업을 도와드립니다. 원하는 작업을 자연어로 설명해주세요!

        **대화 예시:**
        - "유튜브 쇼츠 미스테리 영상을 만들고 싶어"
        - (계획 제시 후) "좋아 진행해" / "2번은 빼줘" / "취소할래"
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="대화",
                    height=500
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="메시지 입력",
                        placeholder="AI 도구 추천을 요청하거나, 계획에 대해 응답하세요...",
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("전송", variant="primary", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### 설정")
                user_id_input = gr.Textbox(
                    label="사용자 ID",
                    value="default_user",
                    placeholder="사용자 ID"
                )
                status_text = gr.Textbox(
                    label="상태",
                    value="대기 중",
                    interactive=False
                )

                gr.Markdown("### 통계")
                with gr.Row():
                    tools_count = gr.Number(label="등록된 도구", value=0, interactive=False)
                    profiles_count = gr.Number(label="사용자 프로필", value=0, interactive=False)

                refresh_btn = gr.Button("새로고침")

        # 이벤트 핸들러 (단일화)
        submit_btn.click(
            fn=process_message,
            inputs=[msg, chatbot, user_id_input, current_thread_id],
            outputs=[chatbot, current_thread_id, status_text]
        ).then(
            fn=lambda: "",
            outputs=msg
        )

        msg.submit(
            fn=process_message,
            inputs=[msg, chatbot, user_id_input, current_thread_id],
            outputs=[chatbot, current_thread_id, status_text]
        ).then(
            fn=lambda: "",
            outputs=msg
        )

        def refresh_stats():
            memory = get_memory_manager()
            return memory.get_tools_count(), memory.get_profiles_count()

        refresh_btn.click(
            fn=refresh_stats,
            outputs=[tools_count, profiles_count]
        )

        # 초기 로드 시 통계 업데이트
        demo.load(
            fn=refresh_stats,
            outputs=[tools_count, profiles_count]
        )

    return demo


# ==================== Gradio Mount ====================

# Gradio 앱 생성
gradio_app = create_gradio_ui()

# FastAPI에 Gradio 마운트
app = gr.mount_gradio_app(app, gradio_app, path="/")


# ==================== 메인 실행 ====================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"\n{'='*50}")
    print(f"AI 101 에이전트 서버 시작")
    print(f"URL: http://localhost:{port}")
    print(f"{'='*50}\n")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
