"""
API Schemas - Pydantic 모델 정의
"""
from typing import Optional, List, Dict
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """채팅 시작 요청"""
    query: str
    user_id: str = "default_user"
    thread_id: Optional[str] = None


class ApproveRequest(BaseModel):
    """계획 승인 요청"""
    thread_id: str
    action: str = "approve"  # approve, modify, cancel
    feedback: Optional[str] = None


class TaskItem(BaseModel):
    """태스크 항목"""
    id: str
    description: str


class ChatResponse(BaseModel):
    """채팅 응답"""
    thread_id: str
    status: str
    message: str
    is_complex: bool = False
    plan: Optional[List[TaskItem]] = None
    final_guide: Optional[str] = None


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    tools_count: int
    profiles_count: int
