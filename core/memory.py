"""
Memory Manager - ChromaDB 기반 벡터 저장소 및 사용자 프로필 관리
"""
import json
import os
import glob
from typing import List, Dict, Optional
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import DB_PATH


class MemoryManager:
    """ChromaDB를 활용한 RAG 및 장기 메모리 관리 클래스"""

    def __init__(self, persist_dir: str = None):
        """
        Args:
            persist_dir: ChromaDB 영구 저장소 경로
        """
        self.persist_dir = persist_dir or DB_PATH

        # ChromaDB 클라이언트 초기화 (Persistent)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # 임베딩 모델 초기화 (다국어 지원)
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )

        # 컬렉션 초기화
        self.tools_collection = self.client.get_or_create_collection(
            name="ai_tools",
            metadata={"description": "AI tools knowledge base"}
        )

        self.profile_collection = self.client.get_or_create_collection(
            name="user_profile",
            metadata={"description": "User preferences and history"}
        )

        # PDF 지식베이스 컬렉션
        self.pdf_collection = self.client.get_or_create_collection(
            name="pdf_knowledge",
            metadata={"description": "PDF documents knowledge base"}
        )

    def _embed_text(self, text: str) -> List[float]:
        """텍스트를 임베딩 벡터로 변환"""
        return self.embedding_model.encode(text).tolist()

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 임베딩 벡터로 변환"""
        return self.embedding_model.encode(texts).tolist()

    # ==================== AI Tools 관련 메서드 ====================

    def load_tools_from_json(self, json_path: str) -> int:
        """
        JSON 파일에서 AI 도구 데이터를 로드하여 ChromaDB에 저장

        Args:
            json_path: tools_base.json 파일 경로

        Returns:
            저장된 도구 수
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tools = data.get('tools', [])
        if not tools:
            return 0

        # 기존 데이터 확인 (중복 방지)
        existing_count = self.tools_collection.count()
        if existing_count > 0:
            print(f"기존 {existing_count}개의 도구 데이터가 있습니다. 스킵합니다.")
            return existing_count

        # 문서 준비
        documents = []
        metadatas = []
        ids = []

        for idx, tool in enumerate(tools):
            # 검색용 문서 텍스트 생성
            features_text = ", ".join(tool.get("features", []))
            tags_text = ", ".join(tool.get("tags", []))

            doc_text = f"""
            도구명: {tool['name']}
            카테고리: {tool['category']}
            설명: {tool['description']}
            기능: {features_text}
            태그: {tags_text}
            가격: {tool['pricing']}
            """.strip()

            documents.append(doc_text)

            # 메타데이터
            metadatas.append({
                "name": tool['name'],
                "category": tool['category'],
                "description": tool['description'],
                "pricing": tool['pricing'],
                "monthly_price": tool.get('monthly_price', 0),
                "url": tool['url'],
                "features": features_text,
                "tags": tags_text,
                "updated_at": tool.get('updated_at', '')
            })

            # ID
            ids.append(f"tool_{idx}_{tool['name'].lower().replace(' ', '_')}")

        # 임베딩 생성 및 저장
        embeddings = self._embed_texts(documents)

        self.tools_collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"{len(tools)}개의 AI 도구 데이터를 로드했습니다.")
        return len(tools)

    def search_tools(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7,
        category: Optional[str] = None
    ) -> tuple[List[Dict], bool]:
        """
        AI 도구 검색 (RAG)

        Args:
            query: 검색 쿼리
            k: 반환할 최대 결과 수
            threshold: 유사도 임계값 (0.7)
            category: 카테고리 필터 (선택)

        Returns:
            (검색 결과 리스트, fallback 필요 여부)
        """
        # 도구 데이터가 없으면 빈 리스트 반환 (HNSW 오류 방지)
        if self.tools_collection.count() == 0:
            return [], True

        # 쿼리 임베딩
        query_embedding = self._embed_text(query)

        # 카테고리 필터
        where_filter = {"category": category} if category else None

        # ChromaDB 검색
        results = self.tools_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter
        )

        # 결과 처리
        search_results = []

        if results['documents'] and results['documents'][0]:
            for idx, doc in enumerate(results['documents'][0]):
                # 거리 → 유사도 변환 (ChromaDB는 L2 거리 사용)
                distance = results['distances'][0][idx] if results['distances'] else 1.0
                # L2 거리를 유사도로 변환: 1 / (1 + distance)
                similarity = 1 / (1 + distance)

                metadata = results['metadatas'][0][idx] if results['metadatas'] else {}

                search_results.append({
                    "name": metadata.get("name", "Unknown"),
                    "category": metadata.get("category", ""),
                    "description": metadata.get("description", ""),
                    "pricing": metadata.get("pricing", ""),
                    "monthly_price": metadata.get("monthly_price", 0),
                    "url": metadata.get("url", ""),
                    "features": metadata.get("features", ""),
                    "tags": metadata.get("tags", ""),
                    "updated_at": metadata.get("updated_at", ""),
                    "score": round(similarity, 3)
                })

        # Fallback 필요 여부 판단
        if not search_results:
            should_fallback = True
        else:
            top_score = max(r['score'] for r in search_results)
            avg_score = sum(r['score'] for r in search_results) / len(search_results)
            should_fallback = top_score < threshold or avg_score < 0.5

        return search_results, should_fallback

    def get_tool_by_name(self, name: str) -> Optional[Dict]:
        """도구 이름으로 상세 정보 조회"""
        results = self.tools_collection.get(
            where={"name": name},
            limit=1
        )

        if results['metadatas']:
            return results['metadatas'][0]
        return None

    # ==================== 사용자 프로필 관련 메서드 ====================

    def load_user_profile(self, user_id: str) -> Optional[Dict]:
        """
        장기 메모리에서 사용자 프로필 로드

        Args:
            user_id: 사용자 ID

        Returns:
            사용자 프로필 딕셔너리 또는 None
        """
        try:
            results = self.profile_collection.get(
                ids=[f"profile_{user_id}"],
                include=["documents", "metadatas"]
            )

            if results['documents'] and results['documents'][0]:
                profile_json = results['documents'][0]
                profile = json.loads(profile_json)
                return profile
        except Exception as e:
            print(f"프로필 로드 실패: {e}")

        return None

    def save_user_profile(self, user_id: str, preferences: Dict) -> bool:
        """
        사용자 프로필을 ChromaDB에 저장 (Upsert)

        Args:
            user_id: 사용자 ID
            preferences: 사용자 선호도 딕셔너리

        Returns:
            저장 성공 여부
        """
        try:
            profile_id = f"profile_{user_id}"
            profile_json = json.dumps(preferences, ensure_ascii=False)

            # 프로필 텍스트를 임베딩 (향후 유사 사용자 검색용)
            profile_text = f"""
            선호 카테고리: {', '.join(preferences.get('preferred_categories', []))}
            선호 가격대: {preferences.get('price_preference', '')}
            관심사: {', '.join(preferences.get('interests', []))}
            기술 수준: {preferences.get('skill_level', '')}
            """.strip()

            embedding = self._embed_text(profile_text)

            # Upsert (있으면 업데이트, 없으면 추가)
            self.profile_collection.upsert(
                ids=[profile_id],
                documents=[profile_json],
                embeddings=[embedding],
                metadatas=[{
                    "user_id": user_id,
                    "updated_at": datetime.now().isoformat()
                }]
            )

            return True
        except Exception as e:
            print(f"프로필 저장 실패: {e}")
            return False

    def extract_preferences(
        self,
        messages: List[Dict[str, str]],
        existing_profile: Optional[Dict] = None
    ) -> Dict:
        """
        대화 내용에서 사용자 선호도 추출 (Reflection)

        Args:
            messages: 대화 메시지 리스트
            existing_profile: 기존 프로필 (있으면 병합)

        Returns:
            추출된 선호도 딕셔너리
        """
        # 대화 내용을 텍스트로 변환
        conversation_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages
        ])

        # LLM으로 선호도 추출
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        system_prompt = """당신은 대화 분석 전문가입니다.
주어진 대화 내용에서 사용자의 AI 도구 관련 선호도를 분석하세요.

다음 JSON 형식으로만 응답하세요:
{
    "preferred_categories": ["카테고리1", "카테고리2"],
    "price_preference": "무료선호/유료가능/비용무관",
    "interests": ["관심분야1", "관심분야2"],
    "skill_level": "초급/중급/고급",
    "notes": "추가 메모"
}

카테고리 옵션: text-generation, image-generation, video-generation, audio-generation, code-generation, productivity, design, research
"""

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"대화 내용:\n{conversation_text}")
            ])

            # JSON 파싱
            response_text = response.content.strip()
            # JSON 블록 추출
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            preferences = json.loads(response_text)

            # 기존 프로필과 병합
            if existing_profile:
                merged = existing_profile.copy()
                for key, value in preferences.items():
                    if isinstance(value, list) and isinstance(merged.get(key), list):
                        # 리스트는 합집합
                        merged[key] = list(set(merged[key] + value))
                    elif value:  # 새 값이 있으면 업데이트
                        merged[key] = value
                return merged

            return preferences

        except Exception as e:
            print(f"선호도 추출 실패: {e}")
            return existing_profile or {
                "preferred_categories": [],
                "price_preference": "비용무관",
                "interests": [],
                "skill_level": "중급",
                "notes": ""
            }

    def get_tools_count(self) -> int:
        """저장된 AI 도구 수 반환"""
        return self.tools_collection.count()

    def get_profiles_count(self) -> int:
        """저장된 사용자 프로필 수 반환"""
        return self.profile_collection.count()

    # ==================== PDF 지식베이스 관련 메서드 ====================

    def load_pdfs_from_directory(self, pdf_dir: str) -> int:
        """
        디렉토리 내 모든 PDF 파일을 로드하여 ChromaDB에 저장

        Args:
            pdf_dir: PDF 파일이 있는 디렉토리 경로

        Returns:
            저장된 청크 수
        """
        # 중복 방지 체크
        existing_count = self.pdf_collection.count()
        if existing_count > 0:
            print(f"기존 {existing_count}개의 PDF 청크가 있습니다. 스킵합니다.")
            return existing_count

        # PDF 파일 탐색
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        if not pdf_files:
            print(f"PDF 파일을 찾을 수 없습니다: {pdf_dir}")
            return 0

        # 텍스트 스플리터 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        documents = []
        metadatas = []
        ids = []

        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()

                for page in pages:
                    page_num = page.metadata.get("page", 0)
                    chunks = text_splitter.split_text(page.page_content)

                    for chunk_idx, chunk in enumerate(chunks):
                        if chunk.strip():
                            documents.append(chunk)
                            metadatas.append({
                                "source": "pdf",
                                "filename": filename,
                                "page": page_num,
                                "chunk_idx": chunk_idx
                            })
                            ids.append(f"pdf_{filename}_{page_num}_{chunk_idx}")

                print(f"PDF 로드 완료: {filename} ({len(pages)} 페이지)")

            except Exception as e:
                print(f"PDF 로드 실패 ({filename}): {e}")
                continue

        if not documents:
            return 0

        # 임베딩 생성 및 저장
        embeddings = self._embed_texts(documents)

        self.pdf_collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"총 {len(documents)}개의 PDF 청크를 로드했습니다.")
        return len(documents)

    def search_pdf_knowledge(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.03
    ) -> List[Dict]:
        """
        PDF 지식베이스 검색

        Args:
            query: 검색 쿼리
            k: 반환할 최대 결과 수
            threshold: 유사도 임계값

        Returns:
            검색 결과 리스트
        """
        # PDF 데이터가 없으면 빈 리스트 반환
        if self.pdf_collection.count() == 0:
            return []

        query_embedding = self._embed_text(query)

        results = self.pdf_collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        search_results = []
        if results['documents'] and results['documents'][0]:
            for idx, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][idx] if results['distances'] else 1.0
                # L2 거리를 유사도로 변환: 1 / (1 + distance)
                similarity = 1 / (1 + distance)

                if similarity >= threshold:
                    metadata = results['metadatas'][0][idx] if results['metadatas'] else {}
                    search_results.append({
                        "content": doc,
                        "source": "pdf",
                        "filename": metadata.get("filename", "Unknown"),
                        "page": metadata.get("page", 0),
                        "score": round(similarity, 3)
                    })

        return search_results

    def get_pdf_count(self) -> int:
        """저장된 PDF 청크 수 반환"""
        return self.pdf_collection.count()


# 메모리 매니저 싱글톤
_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """메모리 매니저 싱글톤 반환"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
