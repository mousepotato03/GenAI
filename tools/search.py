"""
Search Tools - RAG 검색 및 웹 검색 도구
"""
import os
import json
from typing import List, Dict, Optional

from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


# ==================== Google Search ====================

class GoogleSearchTool:
    """Google Custom Search API를 활용한 검색 도구"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self._service = None

    @property
    def is_available(self) -> bool:
        """API 사용 가능 여부"""
        return bool(self.api_key and self.search_engine_id)

    def _get_service(self):
        """Google API 서비스 객체 생성 (지연 로딩)"""
        if self._service is None and self.is_available:
            try:
                from googleapiclient.discovery import build
                self._service = build(
                    "customsearch", "v1",
                    developerKey=self.api_key
                )
            except Exception as e:
                print(f"Google Search API 초기화 실패: {e}")
                return None
        return self._service

    def search(
        self,
        query: str,
        num_results: int = 3,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        Google Custom Search 실행

        Args:
            query: 검색 쿼리
            num_results: 결과 수 (최대 10)
            category: AI 도구 카테고리 (검색어 최적화용)

        Returns:
            검색 결과 리스트
        """
        if not self.is_available:
            print("Google Search API가 설정되지 않았습니다.")
            return []

        service = self._get_service()
        if not service:
            return []

        # 쿼리 최적화
        optimized_query = self._optimize_query(query, category)

        try:
            result = service.cse().list(
                q=optimized_query,
                cx=self.search_engine_id,
                num=min(num_results, 10)
            ).execute()

            search_results = []
            for item in result.get('items', []):
                search_results.append({
                    "name": item.get('title', ''),
                    "description": item.get('snippet', ''),
                    "url": item.get('link', ''),
                    "source": "google_search",
                    "score": 0.5  # Google 검색 결과는 기본 점수 0.5
                })

            return search_results

        except Exception as e:
            print(f"Google Search 오류: {e}")
            return []

    def _optimize_query(self, query: str, category: Optional[str] = None) -> str:
        """검색 쿼리 최적화"""
        optimized = f"{query} AI tool"

        # 카테고리별 키워드 추가
        category_keywords = {
            "text-generation": "chatbot text generation",
            "image-generation": "AI image generator art",
            "video-generation": "AI video creation",
            "audio-generation": "AI voice TTS music",
            "code-generation": "AI coding assistant",
            "productivity": "AI productivity tool",
            "design": "AI design tool",
            "research": "AI research tool"
        }

        if category and category in category_keywords:
            optimized += f" {category_keywords[category]}"

        return optimized


# 싱글톤 인스턴스
google_search = GoogleSearchTool()


def web_search(query: str, category: Optional[str] = None) -> List[Dict]:
    """웹 검색 래퍼 함수"""
    return google_search.search(query, category=category)


# ==================== 통합 검색 함수 ====================

# 점수 가중치 설정
JSON_WEIGHT = 0.7  # JSON 검색 점수 가중치
PDF_WEIGHT = 0.3   # PDF 검색 점수 가중치


def two_stage_search(
    memory_manager,
    query: str,
    num_candidates: int = 5,
    threshold: float = 0.4,
    category: Optional[str] = None,
    use_web_fallback: bool = True
) -> tuple[Optional[Dict], List[Dict], bool]:
    """
    2단계 점수 합산 방식 검색

    1단계: JSON 기반 벡터 유사도 검색 → 후보군 추출 (json_score)
    2단계: 각 후보에 대해 PDF 검색 → 추가 점수 부여 (pdf_score)
    최종: final_score = json_score * 0.7 + pdf_score * 0.3 → 최고 점수 1개 반환

    Args:
        memory_manager: MemoryManager 인스턴스
        query: 검색 쿼리
        num_candidates: 1단계에서 추출할 후보 수
        threshold: 유사도 임계값
        category: 카테고리 필터
        use_web_fallback: 웹 검색 폴백 사용 여부

    Returns:
        (최고 점수 도구, 전체 후보 리스트, fallback 발동 여부)
    """
    # ========== 1단계: JSON 검색으로 후보군 추출 ==========
    rag_results, should_fallback = memory_manager.search_tools(
        query=query,
        k=num_candidates,
        threshold=threshold,
        category=category
    )

    if not rag_results:
        # 후보가 없으면 웹 검색 폴백
        if use_web_fallback and google_search.is_available:
            print(f"RAG 결과 없음, 웹 검색 실행...")
            web_results = google_search.search(query, num_results=3, category=category)
            if web_results:
                # 웹 검색 결과는 점수 계산 없이 첫 번째 반환
                top_result = web_results[0]
                top_result["source"] = "web"
                top_result["scores"] = {
                    "json_score": 0,
                    "pdf_score": 0,
                    "final_score": 0.5  # 웹 검색 기본 점수
                }
                return top_result, web_results, True
        return None, [], True

    # ========== 2단계: 각 후보에 대해 PDF 검색으로 추가 점수 ==========
    candidates = []
    for tool in rag_results:
        tool_name = tool.get("name", "")
        categories = tool.get("categories", "")
        json_score = tool.get("score", 0)

        # PDF에서 해당 도구 관련도 점수 계산
        pdf_score = memory_manager.search_pdf_for_tool(
            tool_name=tool_name,
            categories=categories,
            k=3
        )

        # 최종 점수 계산: JSON 70% + PDF 30%
        final_score = (json_score * JSON_WEIGHT) + (pdf_score * PDF_WEIGHT)

        # 결과에 점수 정보 추가
        tool["source"] = "json"
        tool["scores"] = {
            "json_score": round(json_score, 3),
            "pdf_score": round(pdf_score, 3),
            "final_score": round(final_score, 3)
        }
        candidates.append(tool)

    # ========== 3단계: 최고 점수 도구 선택 ==========
    candidates.sort(key=lambda x: x["scores"]["final_score"], reverse=True)
    top_tool = candidates[0] if candidates else None

    return top_tool, candidates, should_fallback


def hybrid_search(
    memory_manager,
    query: str,
    k: int = 5,
    threshold: float = 0.4,
    category: Optional[str] = None,
    use_web_fallback: bool = True,
    include_pdf: bool = True
) -> tuple[List[Dict], bool]:
    """
    하이브리드 검색 (기존 호환성 유지)

    내부적으로 two_stage_search를 사용하여 2단계 점수 합산 방식 적용

    Args:
        memory_manager: MemoryManager 인스턴스
        query: 검색 쿼리
        k: 반환할 최대 결과 수
        threshold: 유사도 임계값
        category: 카테고리 필터
        use_web_fallback: 웹 검색 폴백 사용 여부
        include_pdf: PDF 지식베이스 검색 포함 여부 (현재는 항상 2단계로 사용)

    Returns:
        (검색 결과 리스트, fallback 발동 여부)
    """
    top_tool, candidates, should_fallback = two_stage_search(
        memory_manager=memory_manager,
        query=query,
        num_candidates=k,
        threshold=threshold,
        category=category,
        use_web_fallback=use_web_fallback
    )

    return candidates[:k], should_fallback


# ==================== RAG 도구 (LLM 바인딩용) ====================

# 메모리 매니저 싱글톤 (지연 로딩)
_memory_manager = None


def _get_memory_manager():
    """메모리 매니저 싱글톤 반환"""
    global _memory_manager
    if _memory_manager is None:
        from core.memory import MemoryManager
        _memory_manager = MemoryManager()
    return _memory_manager


@tool
def retrieve_docs(query: str, category: Optional[str] = None) -> str:
    """
    AI 도구 지식베이스에서 최적의 도구를 검색합니다.
    2단계 점수 합산 방식: JSON(70%) + PDF(30%) 점수로 최고 점수 1개 도구 추천

    Args:
        query: 검색 쿼리 (예: "유튜브 쇼츠 제작 AI", "이미지 생성 도구")
        category: 도구 카테고리 필터 (선택사항)
            - text-generation: 텍스트 생성
            - image-generation: 이미지 생성
            - video-generation: 비디오 생성
            - audio-generation: 음성/음악 생성
            - code-generation: 코드 생성
            - productivity: 생산성 도구
            - design: 디자인 도구
            - research: 리서치 도구

    Returns:
        검색 결과 (JSON 문자열) - 최고 점수 도구 + 후보군 정보
    """
    memory = _get_memory_manager()

    # 2단계 점수 합산 검색
    top_tool, candidates, should_fallback = two_stage_search(
        memory_manager=memory,
        query=query,
        num_candidates=5,
        threshold=0.4,
        category=category,
        use_web_fallback=False  # 웹 검색은 별도 도구로
    )

    # 결과 포맷팅
    if top_tool:
        recommended_tool = {
            "name": top_tool.get('name', ''),
            "description": top_tool.get('description', ''),
            "categories": top_tool.get('categories', ''),
            "domains": top_tool.get('domains', ''),
            "pricing_model": top_tool.get('pricing_model', ''),
            "pricing_notes": top_tool.get('pricing_notes', ''),
            "scores": top_tool.get('scores', {})
        }
    else:
        recommended_tool = None

    # 후보군 요약 (이름과 점수만)
    candidates_summary = [
        {
            "name": c.get('name', ''),
            "final_score": c.get('scores', {}).get('final_score', 0)
        }
        for c in candidates
    ]

    return json.dumps({
        "recommended_tool": recommended_tool,
        "candidates": candidates_summary,
        "should_fallback": should_fallback,
        "scoring_method": "JSON 70% + PDF 30%"
    }, ensure_ascii=False, indent=2)


@tool
def google_search_tool(query: str, num_results: int = 3) -> str:
    """
    Google에서 최신 정보를 검색합니다.
    지식베이스에 없는 최신 AI 도구나 트렌드를 찾을 때 사용합니다.

    Args:
        query: 검색 쿼리 (예: "2024 최신 AI 영상 편집 도구")
        num_results: 반환할 결과 수 (기본 3개, 최대 10개)

    Returns:
        검색 결과 (JSON 문자열) - 제목, 설명, URL 포함
    """
    if not google_search.is_available:
        return json.dumps({
            "status": "error",
            "message": "Google Search API가 설정되지 않았습니다.",
            "results": []
        }, ensure_ascii=False)

    results = google_search.search(query, num_results=min(num_results, 10))

    return json.dumps({
        "status": "success",
        "query": query,
        "results": results,
        "total_count": len(results)
    }, ensure_ascii=False, indent=2)
