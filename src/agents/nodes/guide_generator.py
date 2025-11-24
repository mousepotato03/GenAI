"""
NODE-05: Guide (Parallel Map)

선정된 도구의 사용법 가이드를 생성합니다.
Fallback 모드인 경우 고도화된 프롬프트를 생성합니다.
"""

from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState, SubTask, ToolCandidate


class GuideNode:
    """사용 가이드 생성 노드"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.5):
        """
        Args:
            model: 사용할 LLM 모델명
            temperature: 생성 다양성 (0.0~1.0)
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def __call__(self, state: AgentState) -> Dict:
        """Guide 노드 실행

        각 서브태스크에 대해 사용 가이드 또는 프롬프트를 생성합니다.

        Args:
            state: 현재 그래프 상태 (plan, evaluation_results, fallback_tasks 포함)

        Returns:
            업데이트된 상태 (guides 포함)
        """
        plan = state.get("plan", [])
        evaluation_results = state.get("evaluation_results", {})
        fallback_tasks = state.get("fallback_tasks", [])

        if not plan:
            print("[Guide] 경고: 계획이 비어있습니다.")
            return {"guides": {}}

        guides = {}

        print(f"\n[Guide] {len(plan)}개 서브태스크의 가이드 생성을 시작합니다...")

        # 각 서브태스크별로 가이드 생성
        for i, subtask in enumerate(plan, 1):
            subtask_id = subtask["id"]
            description = subtask["description"]

            print(f"\n[Guide] [{i}/{len(plan)}] '{description}' 가이드 생성 중...")

            if subtask_id in fallback_tasks:
                # Fallback: 고도화 프롬프트 생성
                guide = self._generate_prompt_guide(subtask)
                print(f"[Guide] '{subtask_id}': Fallback 프롬프트 생성 완료")
            else:
                # 일반: 도구 사용 가이드
                candidates = evaluation_results.get(subtask_id, [])
                if candidates:
                    top_tool = candidates[0]  # 1등 도구
                    guide = self._generate_tool_guide(top_tool, subtask)
                    print(f"[Guide] '{subtask_id}': '{top_tool['name']}' 사용 가이드 생성 완료")
                else:
                    # 후보가 없으면 Fallback
                    guide = self._generate_prompt_guide(subtask)
                    print(f"[Guide] '{subtask_id}': 후보 없음, Fallback 프롬프트 생성")

            guides[subtask_id] = guide

        print(f"\n[Guide] 가이드 생성 완료!")

        return {"guides": guides}

    def _generate_tool_guide(self, tool: ToolCandidate, subtask: SubTask) -> str:
        """도구별 Step-by-step 가이드 생성

        Args:
            tool: 추천 도구
            subtask: 서브태스크

        Returns:
            사용 가이드 (Markdown)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 AI 도구 사용법을 초보자에게 쉽게 설명하는 전문가입니다.

주어진 AI 도구의 사용 가이드를 작성하세요.

# 포함 내용
1. 도구 간단 소개
2. 회원가입/접속 방법
3. 주요 기능 위치
4. 단계별 사용법 (3~5단계)
5. 유용한 팁

# 주의사항
- 초보자도 이해할 수 있도록 쉽게 설명
- 구체적인 단계 제시
- Markdown 형식으로 작성
- 너무 길지 않게 (200-300단어)"""),
            ("user", """다음 AI 도구의 초보자용 사용 가이드를 작성하세요:

**도구명**: {tool_name}
**설명**: {description}
**URL**: {url}
**주요 기능**: {features}

**작업 목표**: {task_description}

위 작업을 수행하기 위한 단계별 가이드를 작성하세요.""")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "tool_name": tool["name"],
            "description": tool.get("description", "AI 도구"),
            "url": tool.get("url", ""),
            "features": ", ".join(tool.get("features", [])) if tool.get("features") else "다양한 기능",
            "task_description": subtask["description"]
        })

        return response.content.strip()

    def _generate_prompt_guide(self, subtask: SubTask) -> str:
        """Fallback: ChatGPT/Claude용 고도화 프롬프트 작성

        특화 도구를 찾지 못한 경우, 범용 LLM으로 작업을 수행할 수 있는
        효과적인 프롬프트를 생성합니다.

        Args:
            subtask: 서브태스크

        Returns:
            고도화 프롬프트 (Markdown)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 프롬프트 엔지니어링 전문가입니다.

주어진 작업을 ChatGPT나 Claude 같은 범용 LLM으로 수행하기 위한
상세하고 효과적인 프롬프트를 작성하세요.

# 프롬프트에 포함할 요소
1. 역할 설정 (You are a...)
2. 구체적인 요구사항
3. 출력 형식 지정
4. Few-shot 예시 (가능한 경우)
5. 제약 조건

# 주의사항
- 명확하고 구체적인 지시
- 단계별 사고 유도 (Chain-of-Thought)
- 출력 형식 명시
- Markdown 코드 블록으로 프롬프트 제공"""),
            ("user", """다음 작업을 수행하기 위한 고도화 프롬프트를 작성하세요:

**작업**: {task_description}
**카테고리**: {category}

ChatGPT 또는 Claude에 입력할 프롬프트를 작성하세요.""")
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "task_description": subtask["description"],
            "category": subtask["category"]
        })

        # 가이드 형식으로 포맷팅
        guide = f"""### 💡 범용 LLM 활용 방법

특화 도구를 찾지 못했지만, ChatGPT나 Claude 같은 범용 AI로 작업을 수행할 수 있습니다.

**권장 도구**: ChatGPT (GPT-4), Claude (Sonnet), Gemini

**사용 방법**:
1. 위 도구 중 하나에 접속
2. 아래 프롬프트를 복사하여 입력
3. 결과를 확인하고 필요시 수정 요청

**프롬프트**:

{response.content.strip()}
"""

        return guide
