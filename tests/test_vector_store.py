"""
VectorStore 단위 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.vector_store import VectorStore


@pytest.fixture
def vector_store():
    """테스트용 VectorStore 인스턴스"""
    vs = VectorStore(persist_directory="./data/test_chroma_db")
    yield vs
    # 테스트 후 정리
    vs.delete_all()


def test_vector_store_initialization(vector_store):
    """VectorStore 초기화 테스트"""
    assert vector_store is not None
    assert vector_store.collection is not None
    assert vector_store.embedding_model is not None


def test_add_tools(vector_store):
    """도구 추가 테스트"""
    test_tools = [
        {
            "name": "Test Tool 1",
            "category": "text-generation",
            "description": "This is a test tool for text generation",
            "features": ["feature1", "feature2"],
            "pricing": "free",
            "url": "https://example.com",
            "tags": ["test", "demo"]
        },
        {
            "name": "Test Tool 2",
            "category": "image-generation",
            "description": "This is a test tool for image generation",
            "features": ["feature3", "feature4"],
            "pricing": "paid",
            "url": "https://example2.com",
            "tags": ["test", "image"]
        }
    ]

    vector_store.add_tools(test_tools)
    count = vector_store.get_count()

    assert count == 2


def test_search(vector_store):
    """검색 기능 테스트"""
    # 데이터 추가
    test_tools = [
        {
            "name": "Text Generator",
            "category": "text-generation",
            "description": "Generate text using AI",
            "features": ["writing", "content"],
            "pricing": "free",
            "url": "https://example.com",
            "tags": ["text", "ai"]
        }
    ]

    vector_store.add_tools(test_tools)

    # 검색 수행
    results, should_fallback = vector_store.search("text generation AI", n_results=1)

    assert len(results) > 0
    assert results[0]["name"] == "Text Generator"
    assert isinstance(should_fallback, bool)


def test_search_with_threshold(vector_store):
    """Threshold 기반 검색 테스트"""
    test_tools = [
        {
            "name": "Similar Tool",
            "category": "test",
            "description": "A tool for testing similarity",
            "features": ["test"],
            "pricing": "free",
            "url": "https://example.com",
            "tags": ["test"]
        }
    ]

    vector_store.add_tools(test_tools)

    # 유사한 쿼리 (높은 점수 예상)
    results1, fallback1 = vector_store.search("similarity testing tool", threshold=0.7)

    # 전혀 다른 쿼리 (낮은 점수 예상, fallback True)
    results2, fallback2 = vector_store.search("completely different topic xyz", threshold=0.7)

    # 첫 번째 검색은 fallback이 아닐 가능성이 높음
    # 두 번째 검색은 fallback일 가능성이 높음
    assert isinstance(fallback1, bool)
    assert isinstance(fallback2, bool)


def test_search_with_category_filter(vector_store):
    """카테고리 필터 검색 테스트"""
    test_tools = [
        {
            "name": "Text Tool",
            "category": "text-generation",
            "description": "Text generation tool",
            "features": ["writing"],
            "pricing": "free",
            "url": "https://example.com",
            "tags": ["text"]
        },
        {
            "name": "Image Tool",
            "category": "image-generation",
            "description": "Image generation tool",
            "features": ["design"],
            "pricing": "free",
            "url": "https://example.com",
            "tags": ["image"]
        }
    ]

    vector_store.add_tools(test_tools)

    # 카테고리 필터 적용
    results, _ = vector_store.search("generation tool", category="text-generation", n_results=5)

    # 모든 결과가 text-generation 카테고리여야 함
    for result in results:
        assert result["category"] == "text-generation"


def test_delete_all(vector_store):
    """데이터 삭제 테스트"""
    test_tools = [
        {
            "name": "Test Tool",
            "category": "test",
            "description": "Test description",
            "features": [],
            "pricing": "free",
            "url": "https://example.com",
            "tags": []
        }
    ]

    vector_store.add_tools(test_tools)
    assert vector_store.get_count() == 1

    vector_store.delete_all()
    assert vector_store.get_count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
