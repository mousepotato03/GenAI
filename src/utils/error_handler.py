"""
AI 101 에러 핸들링 모듈

커스텀 예외 클래스 및 전역 에러 핸들러 정의
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Any, Dict
import traceback


# ============================================
# 커스텀 예외 클래스
# ============================================

class AI101Exception(Exception):
    """기본 예외 클래스"""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class APIKeyError(AI101Exception):
    """API 키 관련 에러"""
    pass


class SearchError(AI101Exception):
    """검색 실패 에러"""
    pass


class VectorStoreError(AI101Exception):
    """Vector DB 관련 에러"""
    pass


class PlanningError(AI101Exception):
    """계획 생성 실패 에러"""
    pass


class EvaluationError(AI101Exception):
    """평가 실패 에러"""
    pass


class GraphExecutionError(AI101Exception):
    """LangGraph 실행 에러"""
    pass


class SessionNotFoundError(AI101Exception):
    """세션을 찾을 수 없음"""
    pass


# ============================================
# 에러 응답 생성 함수
# ============================================

def create_error_response(
    error: Exception,
    status_code: int = 500,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    에러 응답 딕셔너리 생성

    Args:
        error: 발생한 예외
        status_code: HTTP 상태 코드
        include_traceback: 트레이스백 포함 여부 (디버그 모드)

    Returns:
        에러 응답 딕셔너리
    """
    response = {
        "error": True,
        "error_type": type(error).__name__,
        "message": str(error)
    }

    # AI101Exception의 경우 추가 정보 포함
    if isinstance(error, AI101Exception):
        response["details"] = error.details

    # 트레이스백 포함 (디버그 모드)
    if include_traceback:
        response["traceback"] = traceback.format_exc()

    return response


# ============================================
# FastAPI 예외 핸들러
# ============================================

async def ai101_exception_handler(request: Request, exc: AI101Exception) -> JSONResponse:
    """
    AI101Exception 전역 핸들러

    Args:
        request: FastAPI 요청 객체
        exc: 발생한 AI101Exception

    Returns:
        JSON 에러 응답
    """
    # 예외 타입별 상태 코드 매핑
    status_code_mapping = {
        APIKeyError: 500,
        SearchError: 503,
        VectorStoreError: 500,
        PlanningError: 422,
        EvaluationError: 500,
        GraphExecutionError: 500,
        SessionNotFoundError: 404,
    }

    status_code = status_code_mapping.get(type(exc), 500)

    # 에러 로깅 (추후 logger 연동 시 사용)
    print(f"[ERROR] {type(exc).__name__}: {exc.message}")
    if exc.details:
        print(f"[ERROR] Details: {exc.details}")

    return JSONResponse(
        status_code=status_code,
        content=create_error_response(exc, status_code)
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    일반 예외 전역 핸들러

    Args:
        request: FastAPI 요청 객체
        exc: 발생한 일반 예외

    Returns:
        JSON 에러 응답
    """
    # 에러 로깅
    print(f"[ERROR] Unhandled exception: {type(exc).__name__}: {str(exc)}")
    print(f"[ERROR] Traceback:\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content=create_error_response(
            exc,
            status_code=500,
            include_traceback=False  # 프로덕션에서는 False
        )
    )


# ============================================
# 에러 래퍼 함수
# ============================================

def wrap_api_error(func):
    """
    API 호출 에러 래핑 데코레이터

    OpenAI API, Google Search API 등의 에러를
    AI101Exception으로 변환
    """
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_message = str(e)

            # API 키 관련 에러
            if "api key" in error_message.lower() or "unauthorized" in error_message.lower():
                raise APIKeyError(
                    f"API 키 오류: {error_message}",
                    details={"original_error": type(e).__name__}
                )

            # 검색 관련 에러
            elif "search" in error_message.lower() or "quota" in error_message.lower():
                raise SearchError(
                    f"검색 오류: {error_message}",
                    details={"original_error": type(e).__name__}
                )

            # 기타 에러는 그대로 전달
            else:
                raise

    return wrapper


def safe_execute(func, default_value=None, error_type: type = AI101Exception):
    """
    함수 실행을 안전하게 래핑

    Args:
        func: 실행할 함수
        default_value: 에러 발생 시 반환할 기본값
        error_type: 발생시킬 에러 타입

    Returns:
        함수 실행 결과 또는 기본값
    """
    try:
        return func()
    except Exception as e:
        print(f"[WARNING] Error in {func.__name__}: {str(e)}")
        if default_value is not None:
            return default_value
        else:
            raise error_type(
                f"함수 실행 실패: {func.__name__}",
                details={"error": str(e)}
            )
