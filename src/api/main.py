"""
FastAPI Backend

AI 101 에이전트 시스템의 REST API 서버입니다.
LangGraph 워크플로우를 실행하고 세션을 관리합니다.
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.graph import create_graph
from src.agents.nodes.memory import load_user_profile
from src.utils.error_handler import (
    AI101Exception,
    ai101_exception_handler,
    general_exception_handler
)

# 환경변수 로드
load_dotenv()

# FastAPI 앱 초기화
app = FastAPI(
    title="AI 101 API",
    description="AI 도구 추천 에이전트 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적인 도메인 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error Handler 등록
app.add_exception_handler(AI101Exception, ai101_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# 세션 저장소 (프로덕션에서는 Redis 등 사용)
sessions = {}

# LangGraph 인스턴스 (재사용)
graph = None


# ===== Request/Response Models =====

class ChatRequest(BaseModel):
    """채팅 요청"""
    query: str
    session_id: Optional[str] = None


class SubTaskInfo(BaseModel):
    """서브태스크 정보"""
    id: str
    description: str
    category: str
    status: str


class ChatResponse(BaseModel):
    """채팅 응답"""
    session_id: str
    status: str  # "planning", "awaiting_approval", "processing", "completed", "error"
    plan: Optional[List[SubTaskInfo]] = None
    response: Optional[str] = None
    error: Optional[str] = None


class ApprovalRequest(BaseModel):
    """계획 승인 요청"""
    session_id: str
    approved: bool
    feedback: Optional[str] = None


# ===== API Endpoints =====

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 그래프 초기화"""
    global graph
    print("\n[API] FastAPI 서버 시작 중...")
    print("[API] LangGraph 초기화 중...")
    graph = create_graph()
    print("[API] LangGraph 초기화 완료!")
    print("[API] 서버 준비 완료!\n")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "AI 101 API 서버",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """메인 대화 엔드포인트

    사용자 질문을 처리하고 계획을 생성합니다.
    계획은 사용자 승인을 기다립니다.

    Args:
        request: 채팅 요청 (query, session_id)

    Returns:
        ChatResponse: 세션 ID, 상태, 계획 등
    """
    try:
        # 1. 세션 생성 또는 로드
        session_id = request.session_id or str(uuid.uuid4())

        print(f"\n[API] 새 요청: session_id={session_id}")
        print(f"[API] 사용자 질문: {request.query}")

        # 2. 사용자 프로필 로드
        user_profile = load_user_profile(user_id=session_id)

        # 3. 초기 상태 생성
        initial_state = {
            "user_query": request.query,
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

        # 4. Config (세션 ID)
        config = {"configurable": {"thread_id": session_id}}

        # 5. 그래프 실행 (interrupt까지)
        print(f"[API] 그래프 실행 시작...")
        result = graph.invoke(initial_state, config)

        # 6. 세션 저장
        sessions[session_id] = {
            "query": request.query,
            "state": result,
            "config": config
        }

        # 7. 현재 상태 확인
        if result.get("error"):
            # 에러 발생
            return ChatResponse(
                session_id=session_id,
                status="error",
                error=result["error"]
            )

        if not result.get("plan_approved", False):
            # Plan 승인 대기
            plan = result.get("plan", [])
            plan_info = [
                SubTaskInfo(
                    id=task["id"],
                    description=task["description"],
                    category=task["category"],
                    status=task["status"]
                )
                for task in plan
            ]

            print(f"[API] Plan 생성 완료, 승인 대기 중")

            return ChatResponse(
                session_id=session_id,
                status="awaiting_approval",
                plan=plan_info
            )

        # 8. 완료 상태 (일반적으로는 여기 도달하지 않음)
        return ChatResponse(
            session_id=session_id,
            status="completed",
            response=result.get("final_response", "")
        )

    except Exception as e:
        print(f"[API] 에러 발생: {e}")
        import traceback
        traceback.print_exc()

        return ChatResponse(
            session_id=session_id if 'session_id' in locals() else "unknown",
            status="error",
            error=str(e)
        )


@app.post("/approve", response_model=ChatResponse)
async def approve_plan(request: ApprovalRequest):
    """Plan 승인/거절 처리

    사용자가 계획을 승인하거나 거절합니다.
    승인 시 그래프를 재개하여 전체 워크플로우를 실행합니다.

    Args:
        request: 승인 요청 (session_id, approved, feedback)

    Returns:
        ChatResponse: 최종 응답 또는 수정된 계획
    """
    try:
        session_id = request.session_id

        # 1. 세션 확인
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

        session = sessions[session_id]
        config = session["config"]

        print(f"\n[API] Plan 승인 요청: session_id={session_id}, approved={request.approved}")

        # 2. 현재 상태 로드
        state = graph.get_state(config)

        if request.approved:
            # 승인: plan_approved = True
            print(f"[API] 계획 승인됨, 그래프 재개 중...")

            graph.update_state(config, {"plan_approved": True})

            # 그래프 재개 (나머지 노드 실행)
            result = graph.invoke(None, config)

            # 세션 업데이트
            sessions[session_id]["state"] = result

            print(f"[API] 워크플로우 완료!")

            return ChatResponse(
                session_id=session_id,
                status="completed",
                response=result.get("final_response", "응답을 생성할 수 없습니다.")
            )

        else:
            # 거절: 피드백 반영하여 Plan 재생성
            print(f"[API] 계획 거절됨, 피드백: {request.feedback}")

            feedback = request.feedback or "다시 생각해주세요"
            graph.update_state(config, {
                "plan_approved": False,
                "user_feedback": feedback
            })

            # Plan 노드 재실행
            result = graph.invoke(None, config)

            # 세션 업데이트
            sessions[session_id]["state"] = result

            # 새 계획 반환
            plan = result.get("plan", [])
            plan_info = [
                SubTaskInfo(
                    id=task["id"],
                    description=task["description"],
                    category=task["category"],
                    status=task["status"]
                )
                for task in plan
            ]

            print(f"[API] 새 계획 생성 완료")

            return ChatResponse(
                session_id=session_id,
                status="awaiting_approval",
                plan=plan_info
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] 에러 발생: {e}")
        import traceback
        traceback.print_exc()

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """세션 상태 조회

    Args:
        session_id: 세션 ID

    Returns:
        세션 정보 (query, state)
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

    session = sessions[session_id]

    return {
        "session_id": session_id,
        "query": session.get("query"),
        "state": {
            "plan_approved": session["state"].get("plan_approved", False),
            "plan_count": len(session["state"].get("plan", [])),
            "final_response_available": bool(session["state"].get("final_response"))
        }
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제

    Args:
        session_id: 세션 ID

    Returns:
        삭제 확인 메시지
    """
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"세션 {session_id}이 삭제되었습니다"}
    else:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")


# ===== 서버 실행 =====
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    print(f"\n{'='*60}")
    print(f"AI 101 API 서버 시작")
    print(f"주소: http://{host}:{port}")
    print(f"문서: http://{host}:{port}/docs")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=host, port=port)
