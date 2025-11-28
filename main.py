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
    get_memory_manager
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
    plan: Optional[list] = None
    final_response: Optional[str] = None


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

        # Plan 단계까지 실행 (human_review 전에 interrupt)
        final_state = None
        for event in graph.stream(initial_state, config):
            for node_name, node_output in event.items():
                print(f"[{thread_id}] Node: {node_name}")
            final_state = event

        # 현재 상태 조회
        state = graph.get_state(config)
        subtasks = state.values.get("subtasks", [])
        plan_analysis = state.values.get("plan_analysis", "")

        # 세션 저장
        active_sessions[thread_id] = {
            "graph": graph,
            "config": config,
            "state": state
        }

        return ChatResponse(
            thread_id=thread_id,
            status="pending_approval",
            message=f"작업 계획을 수립했습니다.\n\n분석: {plan_analysis}\n\n승인하시겠습니까?",
            plan=[{
                "id": t.get("id"),
                "description": t.get("description"),
                "category": t.get("category")
            } for t in subtasks]
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
        # 사용자 응답으로 그래프 재개
        user_response = {
            "action": request.action,
            "feedback": request.feedback or ""
        }

        # Command로 interrupt 재개
        from langgraph.types import Command

        final_state = None
        for event in graph.stream(Command(resume=user_response), config):
            for node_name, node_output in event.items():
                print(f"[{thread_id}] Node: {node_name}")
            final_state = event

        # 최종 상태 조회
        state = graph.get_state(config)
        final_response = state.values.get("final_response", "")

        # 세션 정리
        if thread_id in active_sessions:
            del active_sessions[thread_id]

        if request.action == "cancel":
            return ChatResponse(
                thread_id=thread_id,
                status="cancelled",
                message="작업이 취소되었습니다.",
                final_response=None
            )

        return ChatResponse(
            thread_id=thread_id,
            status="completed",
            message="작업이 완료되었습니다.",
            final_response=final_response
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Gradio UI ====================

def create_gradio_ui():
    """Gradio 채팅 인터페이스 생성"""
    
    def chat_start(message: str, history: list, user_id: str):
        """채팅 시작 - Plan 생성"""
        if not message.strip():
            return history, None, None, "메시지를 입력해주세요."

        thread_id = str(uuid.uuid4())

        try:
            graph = create_agent_graph()
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = create_initial_state(message, user_id or "gradio_user")

            # Plan 단계까지 실행
            for event in graph.stream(initial_state, config):
                pass

            state = graph.get_state(config)
            subtasks = state.values.get("subtasks", [])
            plan_analysis = state.values.get("plan_analysis", "")

            # 세션 저장
            active_sessions[thread_id] = {
                "graph": graph,
                "config": config,
                "state": state
            }

            # 계획 메시지 생성
            plan_text = f"**📋 작업 분석**\n{plan_analysis}\n\n**📝 수립된 계획:**\n"
            for task in subtasks:
                plan_text += f"- **{task['id']}**: {task['description']} ({task['category']})\n"
            plan_text += "\n*아래 버튼으로 승인/취소해주세요.*"

            # 히스토리 업데이트
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": plan_text})

            return history, thread_id, subtasks, "계획 승인 대기 중..."

        except Exception as e:
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})
            return history, None, None, f"오류: {str(e)}"

    def approve_plan(history: list, thread_id: str, feedback: str):
        """계획 승인"""
        if not thread_id or thread_id not in active_sessions:
            return history, None, None, "활성 세션이 없습니다."

        session = active_sessions[thread_id]
        graph = session["graph"]
        config = session["config"]

        try:
            from langgraph.types import Command

            user_response = {"action": "approve", "feedback": feedback or ""}

            # 그래프 재개
            for event in graph.stream(Command(resume=user_response), config):
                pass

            state = graph.get_state(config)
            final_response = state.values.get("final_response", "작업이 완료되었습니다.")

            # 히스토리 업데이트
            history.append({"role": "assistant", "content": f"✅ **계획 승인됨**\n\n{final_response}"})

            # 세션 정리
            if thread_id in active_sessions:
                del active_sessions[thread_id]

            return history, None, None, "완료!"

        except Exception as e:
            history.append({"role": "assistant", "content": f"오류: {str(e)}"})
            return history, None, None, f"오류: {str(e)}"

    def cancel_plan(history: list, thread_id: str):
        """계획 취소"""
        if thread_id and thread_id in active_sessions:
            del active_sessions[thread_id]

        history.append({"role": "assistant", "content": "❌ **작업이 취소되었습니다.**"})
        return history, None, None, "취소됨"

    # UI 구성
    with gr.Blocks(
        title="AI 101 - AI 도구 추천 에이전트"
    ) as demo:
        
        # 상태 변수
        current_thread_id = gr.State(None)
        current_plan = gr.State(None)

        gr.Markdown("""
        # 🤖 AI 101 - 지능형 AI 도구 추천 에이전트

        AI 도구를 활용한 작업을 도와드립니다. 원하는 작업을 자연어로 설명해주세요!

        **예시 질문:**
        - "유튜브 쇼츠 미스테리 영상을 만들고 싶어"
        - "블로그 글을 자동으로 작성하고 싶어"
        - "AI로 로고 디자인을 하고 싶어"
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
                        placeholder="AI 도구 추천을 요청하세요...",
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("전송", variant="primary", scale=1)

                with gr.Row():
                    approve_btn = gr.Button("✅ 계획 승인", variant="primary")
                    cancel_btn = gr.Button("❌ 취소", variant="stop")

                feedback_input = gr.Textbox(
                    label="수정 요청 (선택사항)",
                    placeholder="계획에 대한 수정 요청이 있으면 입력하세요...",
                    visible=True
                )

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 설정")
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

                gr.Markdown("### 📊 통계")
                with gr.Row():
                    tools_count = gr.Number(label="등록된 도구", value=0, interactive=False)
                    profiles_count = gr.Number(label="사용자 프로필", value=0, interactive=False)

                refresh_btn = gr.Button("🔄 새로고침")

        # 이벤트 핸들러
        submit_btn.click(
            fn=chat_start,
            inputs=[msg, chatbot, user_id_input],
            outputs=[chatbot, current_thread_id, current_plan, status_text]
        ).then(
            fn=lambda: "",
            outputs=msg
        )

        msg.submit(
            fn=chat_start,
            inputs=[msg, chatbot, user_id_input],
            outputs=[chatbot, current_thread_id, current_plan, status_text]
        ).then(
            fn=lambda: "",
            outputs=msg
        )

        approve_btn.click(
            fn=approve_plan,
            inputs=[chatbot, current_thread_id, feedback_input],
            outputs=[chatbot, current_thread_id, current_plan, status_text]
        ).then(
            fn=lambda: "",
            outputs=feedback_input
        )

        cancel_btn.click(
            fn=cancel_plan,
            inputs=[chatbot, current_thread_id],
            outputs=[chatbot, current_thread_id, current_plan, status_text]
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
