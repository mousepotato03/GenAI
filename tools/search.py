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

def hybrid_search(
    memory_manager,
    query: str,
    k: int = 5,
    threshold: float = 0.7,
    category: Optional[str] = None,
    use_web_fallback: bool = True,
    include_pdf: bool = True
) -> tuple[List[Dict], bool]:
    """
    하이브리드 검색: RAG (JSON + PDF) + Web Search

    Args:
        memory_manager: MemoryManager 인스턴스
        query: 검색 쿼리
        k: 반환할 최대 결과 수
        threshold: 유사도 임계값
        category: 카테고리 필터
        use_web_fallback: 웹 검색 폴백 사용 여부
        include_pdf: PDF 지식베이스 검색 포함 여부

    Returns:
        (검색 결과 리스트, fallback 발동 여부)
    """
    all_results = []

    # 1. JSON 도구 검색 (ChromaDB - ai_tools 컬렉션)
    rag_results, should_fallback = memory_manager.search_tools(
        query=query,
        k=k,
        threshold=threshold,
        category=category
    )

    # source 표시 추가
    for r in rag_results:
        r["source"] = "json"
    all_results.extend(rag_results)

    # 2. PDF 지식베이스 검색 (ChromaDB - pdf_knowledge 컬렉션)
    if include_pdf:
        pdf_results = memory_manager.search_pdf_knowledge(
            query=query,
            k=3,
            threshold=0.03
        )
        all_results.extend(pdf_results)

    # 3. Fallback 조건 확인 및 웹 검색
    if should_fallback and use_web_fallback and google_search.is_available:
        print(f"RAG 결과 부족 (threshold: {threshold}), 웹 검색 실행...")
        web_results = google_search.search(query, num_results=3, category=category)

        # source 표시 추가
        for wr in web_results:
            wr["source"] = "web"

        # 결과 병합 (중복 제거)
        existing_names = {r.get('name', '').lower() for r in all_results if r.get('name')}
        for wr in web_results:
            if wr.get('name', '').lower() not in existing_names:
                all_results.append(wr)

    # 점수순 정렬
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    return all_results[:k], should_fallback


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
    AI 도구 지식베이스에서 관련 문서를 검색합니다.
    JSON(기본 도구 정보)과 PDF(최신 트렌드)를 하이브리드로 검색합니다.

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
        검색 결과 (JSON 문자열) - 도구명, 설명, 가격, 유사도 점수 포함
    """
    memory = _get_memory_manager()

    results, should_fallback = hybrid_search(
        memory_manager=memory,
        query=query,
        k=5,
        threshold=0.7,
        category=category,
        use_web_fallback=False,  # 웹 검색은 별도 도구로
        include_pdf=True
    )

    # 결과 포맷팅
    formatted_results = []
    for r in results:
        if r.get('source') == 'pdf':
            formatted_results.append({
                "type": "pdf_reference",
                "content": r.get('content', '')[:500],  # 500자 제한
                "filename": r.get('filename', ''),
                "page": r.get('page', 0),
                "score": r.get('score', 0)
            })
        else:
            formatted_results.append({
                "type": "ai_tool",
                "name": r.get('name', ''),
                "category": r.get('category', ''),
                "description": r.get('description', ''),
                "pricing": r.get('pricing', ''),
                "monthly_price": r.get('monthly_price', 0),
                "url": r.get('url', ''),
                "score": r.get('score', 0)
            })

    return json.dumps({
        "results": formatted_results,
        "should_fallback": should_fallback,
        "total_count": len(formatted_results)
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
