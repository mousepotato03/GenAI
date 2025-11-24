"""
NODE-07: Memory Extractor

대화 종료 후 사용자 선호도를 추출하고 저장합니다.
이후 세션에서 개인화된 추천에 활용됩니다.
"""

import json
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState
from src.db.vector_store import VectorStore


class MemoryNode:
    """사용자 선호도 학습 노드"""

    def __init__(self, vector_store: VectorStore = None, model: str = "gpt-4o-mini"):
        """
        Args:
            vector_store: Vector Store 인스턴스 (메모리 저장용)
            model: 사용할 LLM 모델명
        """
        self.llm = ChatOpenAI(model=model, temperature=0.3)
        self.vector_store = vector_store

    def __call__(self, state: AgentState) -> Dict:
        """Memory 노드 실행

        대화 내용을 분석하여 사용자 선호도를 추출하고 저장합니다.
        이 노드는 백그라운드에서 실행되며 워크플로우 진행에 영향을 주지 않습니다.

        Args:
            state: 현재 그래프 상태 (전체 대화 내용 포함)

        Returns:
            빈 딕셔너리 (상태 변경 없음)
        """
        print("\n[Memory] 사용자 선호도 추출 중...")

        try:
            # 1. 대화 내용에서 선호도 추출
            preferences = self._extract_preferences(state)

            # 2. Vector DB 또는 별도 메모리에 저장
            self._save_to_memory(preferences, user_id="default")

            print("[Memory] 선호도 저장 완료!")
            print(f"[Memory] 추출된 선호도: {preferences}")

        except Exception as e:
            print(f"[Memory] 선호도 저장 실패: {e}")

        # 상태 변경 없음 (백그라운드 작업)
        return {}

    def _extract_preferences(self, state: AgentState) -> Dict:
        """사용자 선호도 추출

        대화 내용을 분석하여 다음 정보를 추출:
        - 선호 가격대: 무료/유료/상관없음
        - 관심 카테고리: text, image, video 등
        - 기술 수준: 초급/중급/고급
        - 과거 프로젝트 키워드

        Args:
            state: 현재 상태

        Returns:
            선호도 딕셔너리
        """
        user_query = state.get("user_query", "")
        plan = state.get("plan", [])
        evaluation_results = state.get("evaluation_results", {})

        # 계획 정보 요약
        categories = [task["category"] for task in plan]
        selected_tools = []
        for candidates in evaluation_results.values():
            if candidates:
                selected_tools.append(candidates[0]["name"])

        # 프롬프트 구성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자 행동을 분석하여 선호도를 추출하는 전문가입니다.

대화 내용을 분석하여 사용자의 선호도와 특징을 JSON 형식으로 추출하세요.

# 추출 항목
- preferred_pricing: "무료", "유료", "상관없음"
- interests: 관심 카테고리 목록 (예: ["video", "audio"])
- skill_level: "초급", "중급", "고급"
- project_keywords: 과거 프로젝트 관련 키워드 목록

# 주의사항
- 명시적인 정보가 없으면 "알 수 없음"으로 표시
- interests는 반복된 카테고리를 포함
- project_keywords는 구체적인 키워드만 추출
- 응답은 JSON 형식으로만 반환"""),
            ("user", """다음 대화 내용에서 사용자 선호도를 추출하세요:

**사용자 질문**: {query}

**생성된 계획 카테고리**: {categories}

**선택된 도구들**: {tools}

JSON 형식으로 선호도를 추출하세요.""")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "query": user_query,
            "categories": ", ".join(categories) if categories else "없음",
            "tools": ", ".join(selected_tools) if selected_tools else "없음"
        })

        # JSON 파싱
        try:
            content = response.content.strip()

            # ```json ... ``` 형식이면 추출
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            preferences = json.loads(content)
            return preferences

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Memory] 선호도 파싱 실패: {e}")
            # 기본값 반환
            return {
                "preferred_pricing": "상관없음",
                "interests": categories[:3] if categories else [],
                "skill_level": "초급",
                "project_keywords": []
            }

    def _save_to_memory(self, preferences: Dict, user_id: str):
        """메모리 저장

        현재는 간단히 로그만 출력합니다.
        실제 구현에서는 별도 컬렉션이나 DB에 저장해야 합니다.

        Args:
            preferences: 선호도 딕셔너리
            user_id: 사용자 ID
        """
        # TODO: 실제 저장 로직 구현
        # 옵션 1: ChromaDB의 별도 컬렉션에 저장
        # 옵션 2: SQLite 등 별도 DB 사용
        # 옵션 3: 파일로 저장 (간단한 방법)

        # 현재는 로그만 출력
        print(f"[Memory] 사용자 '{user_id}'의 선호도:")
        print(f"  - 선호 가격대: {preferences.get('preferred_pricing', '알 수 없음')}")
        print(f"  - 관심 분야: {preferences.get('interests', [])}")
        print(f"  - 기술 수준: {preferences.get('skill_level', '알 수 없음')}")
        print(f"  - 프로젝트 키워드: {preferences.get('project_keywords', [])}")

        # 파일로 저장 (간단한 구현)
        try:
            import os
            os.makedirs("data/memory", exist_ok=True)

            memory_file = f"data/memory/{user_id}.json"
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(preferences, f, ensure_ascii=False, indent=2)

            print(f"[Memory] 메모리 파일 저장: {memory_file}")

        except Exception as e:
            print(f"[Memory] 파일 저장 실패: {e}")


def load_user_profile(user_id: str = "default") -> Dict:
    """사용자 프로필 로드

    저장된 선호도를 로드합니다.
    Planner 노드에서 사용됩니다.

    Args:
        user_id: 사용자 ID

    Returns:
        사용자 프로필 (선호도)
    """
    try:
        import os
        memory_file = f"data/memory/{user_id}.json"

        if os.path.exists(memory_file):
            with open(memory_file, "r", encoding="utf-8") as f:
                profile = json.load(f)
            print(f"[Memory] 사용자 프로필 로드: {memory_file}")
            return profile
        else:
            print(f"[Memory] 사용자 프로필 없음")
            return {}

    except Exception as e:
        print(f"[Memory] 프로필 로드 실패: {e}")
        return {}
