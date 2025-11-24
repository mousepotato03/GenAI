"""
LangGraph Agent State Definitions

사용자 쿼리를 처리하는 전체 워크플로우의 상태를 정의합니다.
"""

from typing import TypedDict, List, Dict, Optional
from langgraph.graph import MessagesState


class SubTask(TypedDict):
    """개별 서브태스크

    사용자의 전체 목표를 실행 가능한 단위로 분해한 각 태스크를 나타냅니다.
    """
    id: str                # 고유 ID (예: "task_1", "task_2")
    description: str       # 태스크 설명 (예: "미스테리 스크립트 작성")
    category: str          # 카테고리: text, image, video, audio, code 등
    status: str            # 상태: pending, processing, completed


class ToolCandidate(TypedDict):
    """AI 도구 후보

    각 서브태스크에 대해 검색된 AI 도구 후보를 나타냅니다.
    """
    name: str              # 도구명 (예: "ChatGPT")
    score: float           # Vector 유사도 점수 (0~1)
    description: str       # 도구 설명
    url: str               # 공식 사이트 URL
    category: str          # 카테고리
    pricing: str           # 가격 정보 (무료/유료/프리미엄)
    features: List[str]    # 주요 기능 목록

    # Evaluation 단계에서 추가되는 필드
    reputation_score: Optional[float]  # 평판 점수 (0~1)
    final_score: Optional[float]       # 최종 점수 (0~1)
    pros: Optional[List[str]]          # 장점 목록
    cons: Optional[List[str]]          # 단점 목록


class TaskResult(TypedDict):
    """각 서브태스크의 최종 결과

    각 서브태스크에 대한 추천 도구와 사용 가이드를 포함합니다.
    """
    subtask: SubTask                   # 원본 서브태스크
    recommended_tool: ToolCandidate    # 추천된 도구
    guide: str                         # 사용 가이드 또는 프롬프트
    fallback_mode: bool                # Fallback 모드 여부


class AgentState(MessagesState):
    """전체 그래프 상태

    LangGraph 워크플로우의 전체 상태를 관리합니다.
    MessagesState를 상속하여 메시지 히스토리를 자동으로 관리합니다.
    """
    # ===== 입력 =====
    user_query: str                    # 사용자의 원본 질문
    user_profile: Optional[Dict]       # Memory에서 로드한 사용자 프로필

    # ===== Plan 단계 =====
    plan: List[SubTask]                # 생성된 서브태스크 목록
    plan_approved: bool                # 계획 승인 여부
    user_feedback: Optional[str]       # 사용자 피드백 (거절 시)

    # ===== Research 단계 (Map) =====
    research_results: Dict[str, List[ToolCandidate]]  # subtask_id -> 도구 후보 목록

    # ===== Evaluate 단계 (Map) =====
    evaluation_results: Dict[str, List[ToolCandidate]]  # subtask_id -> 평가된 후보 (정렬됨)

    # ===== Guide 단계 (Map) =====
    guides: Dict[str, str]             # subtask_id -> 사용 가이드

    # ===== Synthesize 단계 (Reduce) =====
    final_response: str                # 최종 Markdown 응답

    # ===== 기타 =====
    fallback_tasks: List[str]          # Fallback이 발동된 서브태스크 ID 목록
    error: Optional[str]               # 에러 메시지 (발생 시)
    retry_count: int                   # 재시도 횟수 (ReAct 루프용)


# 상수 정의
VALID_CATEGORIES = ["text", "image", "video", "audio", "code", "data", "design", "productivity"]
TASK_STATUSES = ["pending", "processing", "completed", "failed"]
PRICING_TYPES = ["무료", "유료", "프리미엄", "오픈소스"]

# ReAct 루프 설정
MAX_RETRIES = 2  # 최대 재시도 횟수 (무한 루프 방지)
