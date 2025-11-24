"""
ChromaDB Vector Store 구현
AI 도구 정보를 저장하고 검색하는 Vector Database 관리 클래스
"""

from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """ChromaDB 기반 Vector Store"""

    def __init__(self, persist_directory: Optional[str] = None):
        """
        ChromaDB 초기화

        Args:
            persist_directory: 데이터 저장 디렉토리 경로
        """
        if persist_directory is None:
            persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Embedding 모델 초기화
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Collection 생성 또는 로드
        self.collection = self.client.get_or_create_collection(
            name="ai_tools",
            metadata={"description": "AI tools knowledge base"}
        )

    def add_tools(self, tools: List[Dict]) -> None:
        """
        AI 도구 정보를 Vector DB에 추가

        Args:
            tools: AI 도구 정보 리스트
                [
                    {
                        "name": "ChatGPT",
                        "category": "text-generation",
                        "description": "...",
                        "features": [...],
                        "pricing": "...",
                        "url": "...",
                        "tags": [...]
                    }
                ]
        """
        if not tools:
            return

        documents = []
        metadatas = []
        ids = []

        for i, tool in enumerate(tools):
            # 검색 가능한 텍스트 생성 (name + description + features)
            features_text = ", ".join(tool.get("features", []))
            tags_text = ", ".join(tool.get("tags", []))
            document = f"{tool['name']}. {tool['description']}. Features: {features_text}. Tags: {tags_text}"

            documents.append(document)

            # 메타데이터 저장
            metadatas.append({
                "name": tool["name"],
                "category": tool.get("category", "general"),
                "pricing": tool.get("pricing", "unknown"),
                "url": tool.get("url", ""),
                "features": features_text,
                "tags": tags_text
            })

            # ID 생성 (이름 기반)
            ids.append(f"tool_{i}_{tool['name'].lower().replace(' ', '_')}")

        # Vector DB에 추가
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        threshold: float = 0.7,
        category: Optional[str] = None
    ) -> Tuple[List[Dict], bool]:
        """
        유사도 검색 (threshold 기반 fallback 포함)

        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 개수
            threshold: 유사도 임계값 (0~1)
            category: 카테고리 필터 (선택사항)

        Returns:
            (results, should_fallback)
            - results: 검색 결과 리스트
            - should_fallback: threshold 이하면 True (Web Search 필요)
        """
        # 카테고리 필터 설정
        where_filter = None
        if category:
            where_filter = {"category": category}

        # Vector 검색 수행
        search_results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )

        # 결과 파싱
        results = []
        should_fallback = True

        if search_results and search_results['metadatas'][0]:
            metadatas = search_results['metadatas'][0]
            distances = search_results['distances'][0]
            documents = search_results['documents'][0]

            for metadata, distance, document in zip(metadatas, distances, documents):
                # 거리를 유사도로 변환 (0에 가까울수록 유사)
                similarity_score = 1 - distance

                result = {
                    "name": metadata.get("name", ""),
                    "category": metadata.get("category", ""),
                    "description": document,
                    "pricing": metadata.get("pricing", ""),
                    "url": metadata.get("url", ""),
                    "features": metadata.get("features", "").split(", ") if metadata.get("features") else [],
                    "tags": metadata.get("tags", "").split(", ") if metadata.get("tags") else [],
                    "score": similarity_score
                }

                results.append(result)

                # Threshold 체크
                if similarity_score >= threshold:
                    should_fallback = False

        return results, should_fallback

    def delete_all(self) -> None:
        """모든 데이터 삭제 (테스트용)"""
        # Collection 삭제
        self.client.delete_collection(name="ai_tools")
        # 다시 생성
        self.collection = self.client.get_or_create_collection(
            name="ai_tools",
            metadata={"description": "AI tools knowledge base"}
        )

    def get_count(self) -> int:
        """저장된 도구 개수 반환"""
        return self.collection.count()
