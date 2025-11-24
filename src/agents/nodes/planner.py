"""
NODE-01: Plan & Refine

사용자의 추상적인 목표를 구체적인 서브태스크로 분해합니다.
사용자 피드백이 있는 경우 계획을 수정합니다.
"""

import json
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState, SubTask


class PlannerNode:
    """사용자 쿼리를 SubTask로 분해하는 노드"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Args:
            model: 사용할 LLM 모델명
            temperature: 생성 다양성 (0.0~1.0, 낮을수록 일관적)
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def __call__(self, state: AgentState) -> Dict:
        """Planner 노드 실행

        Args:
            state: 현재 그래프 상태

        Returns:
            업데이트된 상태 (plan, plan_approved 포함)
        """
        # 1. 사용자 프로필 로드 (Memory)
        profile = state.get("user_profile", {})

        # 2. Plan 생성 또는 수정
        retry_count = state.get("retry_count", 0)

        if state.get("user_feedback"):
            # 사용자 피드백이 있으면 계획 수정
            plan = self._refine_plan(state)
        elif retry_count > 0:
            # ReAct 루프: 재시도 시 이전 결과를 반영한 개선된 계획 생성
            print(f"[Planner] 재시도 {retry_count}회 - 이전 결과를 바탕으로 계획 개선")
            plan = self._retry_plan(state, profile)
        else:
            # 초기 계획 생성
            plan = self._create_plan(state, profile)

        return {
            "plan": plan,
            "plan_approved": False  # 승인 대기 상태
        }

    def _create_plan(self, state: AgentState, profile: Dict) -> List[SubTask]:
        """초기 계획 생성

        Args:
            state: 현재 상태
            profile: 사용자 프로필

        Returns:
            서브태스크 목록
        """
        user_query = state["user_query"]

        # 프롬프트 구성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자의 목표를 실행 가능한 단계로 분해하는 전문가입니다.

사용자의 요청을 분석하여 **2개 이상**의 구체적인 서브태스크로 나누세요.
각 서브태스크는 AI 도구로 해결 가능한 단위여야 합니다.

# 사용자 선호도
{profile}

# 카테고리 목록
- text: 텍스트 생성, 글쓰기, 번역 등
- image: 이미지 생성, 편집, 디자인 등
- video: 영상 생성, 편집, 애니메이션 등
- audio: 음성 생성, 음악 제작, 더빙 등
- code: 코드 작성, 디버깅, 자동화 등
- data: 데이터 분석, 시각화, 처리 등
- design: UI/UX 디자인, 프로토타입 등
- productivity: 생산성 도구, 자동화 등

# Few-shot 예시

**예시 1:**
사용자 질문: "유튜브 미스테리 쇼츠를 만들고 싶어. 시나리오부터 영상, 더빙까지 전부"

분석:
1. 먼저 시나리오를 작성해야 함 → text 카테고리
2. 시나리오를 기반으로 영상 제작 → video 카테고리
3. 영상에 더빙/나레이션 추가 → audio 카테고리

응답 (JSON):
[
  {{
    "id": "task_1",
    "description": "공포 미스테리 스토리 시나리오 작성 (30-60초 분량)",
    "category": "text",
    "status": "pending"
  }},
  {{
    "id": "task_2",
    "description": "시나리오 기반 미스테리 분위기의 쇼츠 영상 생성",
    "category": "video",
    "status": "pending"
  }},
  {{
    "id": "task_3",
    "description": "영상에 어울리는 나레이션/더빙 음성 생성",
    "category": "audio",
    "status": "pending"
  }}
]

**예시 2:**
사용자 질문: "블로그 글을 쓰고 싶은데, 주제는 AI 활용법이야. 썸네일도 만들어줘"

분석:
1. AI 활용법에 대한 블로그 글 작성 → text 카테고리
2. 글에 맞는 썸네일 이미지 생성 → image 카테고리

응답 (JSON):
[
  {{
    "id": "task_1",
    "description": "AI 활용법을 주제로 한 블로그 포스트 작성 (1000-1500자)",
    "category": "text",
    "status": "pending"
  }},
  {{
    "id": "task_2",
    "description": "블로그 포스트에 어울리는 AI 테마 썸네일 이미지 생성",
    "category": "image",
    "status": "pending"
  }}
]

# 주의사항
- 최소 2개 이상의 서브태스크로 분해
- 각 태스크는 명확하고 구체적이어야 함
- 실행 순서를 고려하여 배열 순서 결정
- 적절한 카테고리 선택
- 응답은 반드시 JSON 배열 형식으로만 반환
- id는 "task_1", "task_2" 형식으로 순차 부여
- status는 항상 "pending"으로 설정

이제 사용자의 요청을 분석하고 서브태스크로 분해하세요."""),
            ("user", "{query}")
        ])

        # LLM 호출
        chain = prompt | self.llm
        profile_text = self._format_profile(profile)
        response = chain.invoke({
            "query": user_query,
            "profile": profile_text
        })

        # JSON 파싱
        try:
            # LLM 응답에서 JSON 부분만 추출
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

            # JSON 파싱
            plan_data = json.loads(content)

            # SubTask 형식으로 변환
            plan = []
            for task in plan_data:
                plan.append(SubTask(
                    id=task["id"],
                    description=task["description"],
                    category=task["category"],
                    status="pending"
                ))

            return plan

        except (json.JSONDecodeError, KeyError) as e:
            # 파싱 실패 시 기본 계획 반환
            print(f"[Planner] JSON 파싱 실패: {e}")
            print(f"[Planner] LLM 응답: {response.content}")

            return [
                SubTask(
                    id="task_1",
                    description=f"{user_query}에 대한 솔루션 찾기",
                    category="text",
                    status="pending"
                )
            ]

    def _refine_plan(self, state: AgentState) -> List[SubTask]:
        """사용자 피드백 기반 계획 수정

        Args:
            state: 현재 상태 (기존 plan + user_feedback 포함)

        Returns:
            수정된 서브태스크 목록
        """
        user_query = state["user_query"]
        current_plan = state["plan"]
        feedback = state["user_feedback"]

        # 프롬프트 구성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자의 피드백을 반영하여 계획을 수정하는 전문가입니다.

기존 계획과 사용자 피드백을 분석하여 개선된 계획을 제시하세요.

# 카테고리 목록
- text, image, video, audio, code, data, design, productivity

# 주의사항
- 사용자 피드백을 정확히 반영
- 기존 계획의 좋은 부분은 유지
- 응답은 JSON 배열 형식으로만 반환
- id는 "task_1", "task_2" 형식으로 재부여
- status는 항상 "pending"으로 설정
"""),
            ("user", """원래 질문: {query}

기존 계획:
{current_plan}

사용자 피드백:
{feedback}

위 피드백을 반영하여 개선된 계획을 JSON 배열로 제시하세요.""")
        ])

        # 기존 계획 포맷팅
        current_plan_text = json.dumps(
            [dict(task) for task in current_plan],
            ensure_ascii=False,
            indent=2
        )

        # LLM 호출
        chain = prompt | self.llm
        response = chain.invoke({
            "query": user_query,
            "current_plan": current_plan_text,
            "feedback": feedback
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

            # JSON 파싱
            plan_data = json.loads(content)

            # SubTask 형식으로 변환
            plan = []
            for task in plan_data:
                plan.append(SubTask(
                    id=task["id"],
                    description=task["description"],
                    category=task["category"],
                    status="pending"
                ))

            return plan

        except (json.JSONDecodeError, KeyError) as e:
            # 파싱 실패 시 기존 계획 반환
            print(f"[Planner] Refine 실패: {e}")
            print(f"[Planner] LLM 응답: {response.content}")
            return current_plan

    def _retry_plan(self, state: AgentState, profile: Dict) -> List[SubTask]:
        """재시도 시 이전 결과를 반영한 개선된 계획 생성 (ReAct 루프)

        Args:
            state: 현재 상태 (이전 plan, evaluation_results 포함)
            profile: 사용자 프로필

        Returns:
            개선된 서브태스크 목록
        """
        user_query = state["user_query"]
        previous_plan = state.get("plan", [])
        evaluation_results = state.get("evaluation_results", {})
        retry_count = state.get("retry_count", 0)

        # 이전 시도에서 문제가 있었던 태스크 분석
        problematic_tasks = []
        for subtask in previous_plan:
            task_id = subtask["id"]
            candidates = evaluation_results.get(task_id, [])
            if not candidates or (candidates and candidates[0].get("final_score", 0) < 0.6):
                problematic_tasks.append(subtask)

        # 프롬프트 구성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 이전 시도의 결과를 분석하여 더 나은 계획을 제시하는 전문가입니다.

**상황**: 이전 계획으로 적절한 AI 도구를 찾지 못했습니다. 다른 접근 방식을 시도해야 합니다.

# 개선 전략
1. **더 세분화**: 큰 태스크를 더 작은 단위로 나누기
2. **카테고리 변경**: 다른 카테고리의 도구로 같은 목표 달성 가능한지 검토
3. **대안 접근**: 같은 목표를 다른 방식으로 달성하는 방법 고려

# 카테고리 목록
- text, image, video, audio, code, data, design, productivity

# 주의사항
- 이전 계획과는 **다른 접근 방식** 시도
- 문제가 있었던 태스크는 특히 재검토
- 응답은 JSON 배열 형식으로만 반환
- id는 "task_1", "task_2" 형식으로 재부여
- status는 항상 "pending"으로 설정"""),
            ("user", """원래 질문: {query}

이전 계획 (문제 발생):
{previous_plan}

문제가 있었던 태스크:
{problematic_tasks}

재시도 횟수: {retry_count}

위 정보를 바탕으로 **다른 접근 방식**을 사용한 개선된 계획을 JSON 배열로 제시하세요.""")
        ])

        # 이전 계획 포맷팅
        previous_plan_text = json.dumps(
            [dict(task) for task in previous_plan],
            ensure_ascii=False,
            indent=2
        )

        problematic_tasks_text = json.dumps(
            [dict(task) for task in problematic_tasks],
            ensure_ascii=False,
            indent=2
        ) if problematic_tasks else "없음 (전반적으로 점수가 낮음)"

        # LLM 호출
        chain = prompt | self.llm
        response = chain.invoke({
            "query": user_query,
            "previous_plan": previous_plan_text,
            "problematic_tasks": problematic_tasks_text,
            "retry_count": retry_count
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

            # JSON 파싱
            plan_data = json.loads(content)

            # SubTask 형식으로 변환
            plan = []
            for task in plan_data:
                plan.append(SubTask(
                    id=task["id"],
                    description=task["description"],
                    category=task["category"],
                    status="pending"
                ))

            print(f"[Planner] 재시도 계획 생성 완료: {len(plan)}개 태스크")
            return plan

        except (json.JSONDecodeError, KeyError) as e:
            # 파싱 실패 시 이전 계획 반환
            print(f"[Planner] Retry 계획 파싱 실패: {e}")
            print(f"[Planner] LLM 응답: {response.content}")
            return previous_plan

    def _format_profile(self, profile: Dict) -> str:
        """사용자 프로필을 텍스트로 포맷팅

        Args:
            profile: 사용자 프로필 딕셔너리

        Returns:
            포맷팅된 프로필 텍스트
        """
        if not profile:
            return "신규 사용자 (선호도 정보 없음)"

        lines = []
        if "preferred_pricing" in profile:
            lines.append(f"- 선호 가격대: {profile['preferred_pricing']}")
        if "interests" in profile:
            lines.append(f"- 관심 분야: {', '.join(profile['interests'])}")
        if "skill_level" in profile:
            lines.append(f"- 기술 수준: {profile['skill_level']}")

        return "\n".join(lines) if lines else "선호도 정보 없음"
