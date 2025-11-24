"""
Hybrid Search 시스템
RAG (Vector Search) + Web Search 를 결합한 하이브리드 검색
"""

from typing import List, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
from dotenv import load_dotenv

load_dotenv()


class HybridSearch:
    """RAG + Web Search 하이브리드 검색 시스템"""

    def __init__(self, vector_store, google_api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        """
        Hybrid Search 초기화

        Args:
            vector_store: VectorStore 인스턴스
            google_api_key: Google API 키
            search_engine_id: Google Search Engine ID
        """
        self.vector_store = vector_store

        # Google Custom Search API 설정
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")

        if self.google_api_key and self.search_engine_id:
            try:
                self.google_service = build("customsearch", "v1", developerKey=self.google_api_key)
            except Exception as e:
                print(f"Warning: Google Custom Search API 초기화 실패: {e}")
                self.google_service = None
        else:
            self.google_service = None
            print("Warning: Google API 키 또는 Search Engine ID가 설정되지 않았습니다.")

    def search(self, query: str, category: Optional[str] = None) -> List[Dict]:
        """
        Hybrid Retrieval: RAG → Web Search

        1. Vector DB에서 검색 (threshold 기반)
        2. 결과 부족 시 Google Custom Search 실행
        3. 최대 5개 후보 반환

        Args:
            query: 검색 쿼리
            category: 카테고리 필터 (선택사항)

        Returns:
            검색 결과 리스트 (최대 5개)
        """
        results = []

        # 1. RAG 검색
        rag_results, should_fallback = self.vector_store.search(
            query=query,
            n_results=5,
            threshold=0.7,
            category=category
        )

        results.extend(rag_results)

        # 2. Fallback: Web Search
        if should_fallback or len(results) < 3:
            print(f"Fallback 발동: Web Search 실행 (현재 결과 {len(results)}개)")
            web_results = self._web_search(query, category)
            results.extend(web_results)

        # 3. 중복 제거 (이름 기반)
        seen_names = set()
        unique_results = []
        for result in results:
            if result["name"] not in seen_names:
                seen_names.add(result["name"])
                unique_results.append(result)

        # 4. 점수 순 정렬 및 상위 5개 반환
        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return unique_results[:5]

    def _web_search(self, query: str, category: Optional[str] = None) -> List[Dict]:
        """
        Google Custom Search API 호출

        Args:
            query: 검색 쿼리
            category: 카테고리 (선택사항)

        Returns:
            웹 검색 결과 리스트
        """
        if not self.google_service:
            print("Warning: Google Custom Search API가 사용 불가능합니다.")
            return []

        try:
            # 검색 쿼리 최적화
            optimized_query = self._optimize_query(query, category)

            print(f"Web Search 쿼리: {optimized_query}")

            # Google Custom Search API 호출
            result = self.google_service.cse().list(
                q=optimized_query,
                cx=self.search_engine_id,
                num=3  # 최대 3개 결과
            ).execute()

            # 결과 파싱
            web_results = []
            if 'items' in result:
                for item in result['items']:
                    web_result = {
                        "name": item.get('title', ''),
                        "category": category or "general",
                        "description": item.get('snippet', ''),
                        "pricing": "unknown",
                        "url": item.get('link', ''),
                        "features": [],
                        "tags": [],
                        "score": 0.5,  # Web 검색 결과는 중간 점수
                        "source": "web_search"
                    }
                    web_results.append(web_result)

            return web_results

        except HttpError as e:
            print(f"Google Custom Search API 오류: {e}")
            return []
        except Exception as e:
            print(f"Web Search 오류: {e}")
            return []

    def _optimize_query(self, query: str, category: Optional[str] = None) -> str:
        """
        검색 쿼리 최적화

        Args:
            query: 원본 쿼리
            category: 카테고리

        Returns:
            최적화된 검색 쿼리

        예:
            "미스테리 쇼츠 만들어줘" → "AI mystery short video creation tool"
        """
        # 기본적으로 "AI tool" 추가
        optimized = f"{query} AI tool"

        # 카테고리별 키워드 추가
        category_keywords = {
            "text": "text generation writing",
            "text-generation": "text generation writing",
            "image": "image generation creation",
            "image-generation": "image generation creation",
            "video": "video generation creation",
            "video-generation": "video generation creation",
            "audio": "audio generation voice",
            "audio-generation": "audio generation voice",
            "code": "code generation programming",
            "code-generation": "code generation programming"
        }

        if category and category in category_keywords:
            optimized += f" {category_keywords[category]}"

        return optimized
