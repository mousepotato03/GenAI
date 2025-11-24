# PRD: AI 101 (Intelligent AI Tool Recommendation Agent)

## 프로젝트명: AI 101

## 1. 프로젝트 개요 (Overview)

### 1.1 배경 및 목적

현재 AI 툴 시장은 폭발적으로 성장하고 있으나, 사용자는 자신의 목적에 맞는 최적의 도구를 찾고 검증하는 데 많은 시간을 소모합니다.
**AI 101**은 사용자의 추상적인 아이디어나 요구사항을 분석하여 실행 가능한 단위 작업(Sub-task)으로 분해하고, **최신 트렌드와 평판이 검증된 AI 도구를 추천 및 가이드**해주는 지능형 에이전트 서비스입니다.

### 1.2 프로젝트 목표

1.  **추상적 요구의 구체화:** 사용자의 모호한 요청을 ReAct(Reasoning+Acting) 방식을 통해 논리적인 단계로 분해.
2.  **신뢰 기반 추천:** 단순 검색이 아닌 커뮤니티(Reddit 등) 평판 조회를 통한 검증된 툴 추천.
3.  **실행 가이드 제공:** 도구 추천을 넘어, 실제 사용 가능한 튜토리얼 및 프롬프트 제공.

---

## 2 사용자 스토리 (User Story)

- "나는 '미스테리 유튜브 쇼츠'를 만들고 싶지만, 시나리오부터 영상, 더빙까지 어떤 AI를 써야 할지 몰라 막막하다."
- "나는 검색된 AI 툴이 실제로 쓸만한지, 광고인지, 사용하기 너무 어렵지는 않은지 미리 검증받고 싶다."

---

## 3. 시스템 아키텍처 및 워크플로우 (System Architecture)

본 시스템은 **LangGraph**를 기반으로 **Map-Reduce 패턴**과 **Human-in-the-loop** 프로세스를 결합하여 구축됩니다.

### 3.1 핵심 메커니즘: ReAct & Loop

에이전트는 **[Reasoning(생각) → Acting(행동/검색) → Observation(관찰/평가)]**의 루프를 수행하며, 복잡한 과제를 해결합니다.

### 3.2 워크플로우 요약

1.  **Plan (기획):** 사용자 의도 파악 및 Sub-task 분해.
2.  **Review (검토):** 사용자의 중간 승인 (Human-in-the-loop).
3.  **Research (조사 - Map):** 각 Sub-task별 병렬 검색 수행 (RAG + Web Search).
4.  **Evaluate (평가 - Map):** 후보 툴에 대한 평판 조회 및 점수 산정.
5.  **Guide (가이드 - Map):** 선정된 툴에 대한 사용법 생성.
6.  **Synthesize (통합 - Reduce):** 모든 결과를 취합하여 최종 답변 생성.

---

## 4. 상세 기능 요구사항 (Functional Requirements)

LangGraph의 Node 구조에 따라 기능을 정의합니다.

### 4.1 [NODE-01] Plan & Refine (기획 및 수정)

- **기능:** 사용자 입력(Query)과 사용자 프로필(Memory)을 분석하여 실행 계획(Plan) 수립.
- **세부 요건:**
  - 추상적 목표를 최소 2개 이상의 `SubTask`로 분해해야 함.
  - 사용자 피드백이 있을 경우, 기존 `Plan`을 수정(Update)하는 로직 포함.
- **Output:** `Plan` 객체 (List of SubTasks).

### 4.2 [NODE-02] Human Review (사용자 검토)

- **기능:** 수립된 계획을 사용자에게 제시하고 승인/수정/거절을 받음.
- **세부 요건:**
  - LangGraph의 `interrupt` 기능을 활용하여 실행을 일시 중지.
  - 사용자 승인 시 다음 단계(Map)로 진행, 거절/수정 요청 시 NODE-01로 회귀.

### 4.3 [NODE-03] Research (조사 - Parallel Map)

- **기능:** 각 `SubTask`에 적합한 AI 도구 후보군 탐색.
- **세부 요건:**
  - **Hybrid Retrieval:** 내부 지식 베이스(RAG) 검색 우선 수행 후, 부족 시 웹 검색(Google Custom Search API) 실행.
  - 5개의 후보군(Candidate List) 확보.
  - **Fallback 조건:** 벡터 유사도 검색을 했을 때 점수가 너무 낮으면, 툴 추천을 중단하고 **"대체 프롬프트 제공 모드"**로 플래그 설정.

### 4.5 [NODE-05] Guide (가이드 작성 - Parallel Map)

- **기능:** 선정된 도구의 사용법 가이드 생성.
- **세부 요건:**
  - **Fallback 미발동 시:** 해당 툴의 튜토리얼, 주요 기능 위치, 초보자용 Step-by-step 가이드 작성.
  - **Fallback 발동 시:** ChatGPT/Claude 등 범용 LLM에서 해당 작업을 수행할 수 있는 **고도화된 프롬프트** 작성.

### 4.6 [NODE-06] Synthesize (종합 - Reduce)

- **기능:** 병렬 처리된 결과물들을 하나의 일관된 답변으로 통합.
- **세부 요건:**
  - Markdown 포맷 지원.
  - 전체 워크플로우(서론) -> 단계별 추천 및 가이드(본론) -> 마무리(결론) 구조.

### 4.7 [NODE-07] Memory Extractor (메모리 저장)

- **기능:** 대화 종료 후 백그라운드에서 중요 정보 추출 및 저장.
- **세부 요건:**
  - 사용자 선호도(예: "무료 툴 선호", "영상 제작 관심") 및 과거 프로젝트 이력 추출.
  - Vector DB 또는 Graph DB에 저장하여 다음 세션(NODE-01)에서 활용.

---

## 5. 데이터 및 알고리즘 전략 (Data & Algorithm Strategy)

### 5.1 검색 및 RAG 전략

- **도구 DB 구축:** 주요 AI 도구(ChatGPT, Midjourney, ElevenLabs 등)의 공식 문서 및 특징을 벡터 인덱싱.
- **검색 쿼리 최적화:** 사용자 질문을 검색에 용이한 키워드로 변환 ("미스테리 쇼츠 만들어줘" -> "mystery script writing AI tool", "text to video AI horror style").

## 6. 기술 스택 (Tech Stack)

- **Framework:** LangChain, LangGraph
- **LLM:** GPT-4o-mini
- **Search Tool:** Google Custom Search API
- **Database:** ChromaDB
- **Backend:** Python (FastAPI)

---

## 7. UI/UX 요구사항 (Draft)

Gradio 사용해서 간단한 UI 제작
