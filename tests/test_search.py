"""
HybridSearch 단위 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.vector_store import VectorStore
from src.tools.search import HybridSearch


@pytest.fixture
def vector_store():
    """테스트용 VectorStore 인스턴스"""
    vs = VectorStore(persist_directory="./data/test_chroma_db")

    # 테스트 데이터 추가
    test_tools = [
        {
            "name": "ChatGPT",
            "category": "text-generation",
            "description": "AI chatbot for text generation and conversations",
            "features": ["chat", "text", "writing"],
            "pricing": "free",
            "url": "https://chat.openai.com",
            "tags": ["chatbot", "ai", "text"]
        },
        {
            "name": "DALL-E",
            "category": "image-generation",
            "description": "AI image generation from text prompts",
            "features": ["image", "art", "creative"],
            "pricing": "paid",
            "url": "https://openai.com/dall-e",
            "tags": ["image", "art", "generation"]
        }
    ]

    vs.add_tools(test_tools)
    yield vs

    # 테스트 후 정리
    vs.delete_all()


@pytest.fixture
def hybrid_search(vector_store):
    """테스트용 HybridSearch 인스턴스"""
    # API 키 없이도 테스트 가능 (Web search는 건너뜀)
    return HybridSearch(vector_store, google_api_key=None, search_engine_id=None)


def test_hybrid_search_initialization(hybrid_search):
    """HybridSearch 초기화 테스트"""
    assert hybrid_search is not None
    assert hybrid_search.vector_store is not None


def test_search_with_good_results(hybrid_search):
    """충분한 결과가 있을 때 검색 테스트"""
    results = hybrid_search.search("text generation chatbot")

    assert len(results) > 0
    # ChatGPT가 첫 번째 결과로 나와야 함
    assert results[0]["name"] == "ChatGPT"


def test_search_with_category(hybrid_search):
    """카테고리 필터 검색 테스트"""
    results = hybrid_search.search("generation AI", category="image-generation")

    assert len(results) > 0
    # 이미지 생성 도구만 나와야 함
    assert all(r["category"] == "image-generation" for r in results)


def test_query_optimization(hybrid_search):
    """쿼리 최적화 테스트"""
    optimized1 = hybrid_search._optimize_query("generate text", "text-generation")
    assert "AI tool" in optimized1
    assert "text generation writing" in optimized1

    optimized2 = hybrid_search._optimize_query("create image", "image-generation")
    assert "AI tool" in optimized2
    assert "image generation creation" in optimized2


def test_search_deduplication(hybrid_search):
    """중복 제거 테스트"""
    results = hybrid_search.search("chatbot text generation")

    # 중복 제거 확인
    names = [r["name"] for r in results]
    assert len(names) == len(set(names)), "중복된 결과가 있습니다"


def test_search_max_results(hybrid_search):
    """최대 결과 개수 테스트"""
    results = hybrid_search.search("AI tool")

    # 최대 5개까지만 반환해야 함
    assert len(results) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
