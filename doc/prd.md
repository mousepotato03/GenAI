# PRD: AI 101 (Intelligent AI Tool Recommendation Agent)

## 1. 프로젝트 개요 (Overview)

### 1.1 프로젝트 명

**AI 101** - LangGraph 기반의 지능형 AI 도구 추천 및 가이드 ReAct 에이전트

### 1.2 배경 및 목적

사용자의 추상적인 아이디어(예: "유튜브 쇼츠 제작")를 실현 가능한 단위 작업(Sub-task)으로 분해하고, **검증된 AI 도구(Tool)**를 추천하거나, 도구가 필요 없는 경우 **실질적인 수행 방법(General Advice)**을 가이드해주는 지능형 에이전트 서비스입니다.

### 1.3 핵심 목표 (Final Project 요구사항 반영)

1.  **ReAct 에이전트 구현:** 스스로 사고(Reasoning)하고, 필요한 도구를 호출(Acting)하며, 결과를 관찰(Observation)하는 LangGraph 기반 에이전트 구축.
2.  **Hybrid RAG 시스템:** 정적 데이터(JSON)와 최신 트렌드(PDF/SNS)를 결합한 이원화된 검색 및 검증 시스템.
3.  **지능형 Fallback:** 억지 추천을 지양하고, 적합한 도구가 없을 경우 일반적인 조언 모드로 자동 전환하는 신뢰성 확보.
4.  **메모리 및 개인화:** 사용자의 선호도를 기억(Persistent DB)하여 다음 대화에 반영하는 Long-term Memory 구현.

---

## 2. 필수 요구사항 준수 현황 (Compliance Matrix)

| 구분       | PDF 상세 요구사항                                                | 프로젝트 구현 전략                                                                                                                    |
| :--------- | :--------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **LLM**    | `gpt-4o-mini`, OpenAI API 사용                                   | Main LLM으로 `gpt-4o-mini` 활용.                                                                                                      |
| **Tools**  | 구글 검색, 계산기, 시간                                          | **(1) Search:** Google Custom Search API<br>**(2) Calculator:** 도구 구독료 합산<br>**(3) Time:** 현재 시각 기준 최신 여부 판단       |
| **RAG**    | PDF 색인, Reader, Text Splitter,<br>한영 Embedder, Persistent DB | **JSON(기본) + PDF(최신 뉴스)** 하이브리드 RAG 구축.<br>HuggingFace 다국어 임베더 사용.<br>ChromaDB를 로컬 디스크에 저장(Persistent). |
| **Memory** | Short Term(State),<br>Long Term(Persistent),<br>Reflection       | LangGraph `State`로 대화 문맥 유지.<br>대화 종료 시 사용자 성향 요약(Reflection) 후 DB 저장.<br>다음 세션 시작 시 해당 정보 로드.     |
| **Graph**  | LangGraph, State(Annotated),<br>Interrupt, Stream                | Human-in-the-loop(중간 검토)를 위한 `interrupt`.<br>토큰 단위 스트리밍 응답 구현.                                                     |
| **UI**     | Gradio 사용, **FastAPI Mount**                                   | FastAPI 서버 내에 Gradio를 마운트하여 단일 엔드포인트 제공.                                                                           |

---

## 3. 시스템 아키텍처 및 데이터 흐름

### 3.1 워크플로우 (LangGraph Nodes)

`User Input` → `Long-term Memory Load` → **[Agent ReAct Loop]** → `Output` → `Reflection(Save)`

#### **[Agent ReAct Loop 상세]**

1.  **Plan:** 사용자 의도 파악 및 Sub-task 분해.
2.  **Human Review:** 계획 승인/수정 (Interrupt).
3.  **Research (Hybrid RAG):** JSON(기본) + PDF(최신) 검색 수행.
4.  **Evaluate (Threshold Check):** 검색 결과의 유사도 점수 평가.
    - _Score > Threshold:_ **도구 추천 모드** 진입.
    - _Score < Threshold:_ **일반 조언(Fallback) 모드** 진입.
5.  **Guide:** 모드에 따른 가이드 작성.
6.  **Synthesize:** 최종 답변 통합 생성.

---

## 4. 상세 기능 명세 (Functional Requirements)

### 4.1 [Node-01] Plan & Refine (기획)

- **기능:** 사용자 입력과 장기 기억(Memory)을 바탕으로 실행 계획 수립.
- **Output:** 작업 목록(List of SubTasks). 예: `["대본 작성", "영상 생성", "더빙"]`

### 4.2 [Node-02] Human Review (사용자 검토)

- **기능:** 수립된 계획을 사용자에게 제시하고 진행 여부를 묻음.
- **기술:** LangGraph `interrupt_before` 사용. Gradio UI에서 "승인" 버튼 클릭 시 진행.

### 4.3 [Node-03] Research (조사 및 검증 - 핵심)

- **Data Source 이원화:**
  - **Source A (Base - JSON):** 도구명, 가격, 공식 URL, 기본 기능 등 정형 데이터.
  - **Source B (Updates - PDF):** SNS(X, LinkedIn) 발표 내용, 뉴스레터 등 비정형 최신 데이터.
- **Logic (Retrieval & Score Check):**
  1.  쿼리에 대해 Source A, B 동시 검색.
  2.  검색된 상위 문서의 유사도(Similarity Score) 확인.
  3.  **임계값(Threshold, 예: 0.7) 미만**일 경우 `tool_found = False` 플래그 설정.
  4.  **임계값 이상**일 경우 `tool_found = True` 및 도구 정보 전달.

### 4.4 [Node-04] Tool Execution (필수 도구)

- **Calculator:** 사용자가 예산 견적을 물을 때, 추천된 도구들의 월 구독료 합계 계산.
- **Time:** "최신 정보야?" 질문 시 현재 날짜(datetime)와 업데이트 날짜 비교.

### 4.5 [Node-05] Guide (가이드 작성 - 조건부 생성)

- **기능:** `tool_found` 플래그에 따라 다른 프롬프트 전략 수행.
- **Case A (도구 추천):**
  - 검색된 도구의 구체적 사용법, 링크, 가격 정보 제공.
  - 예: _"Sora를 사용해보세요. 월 $20이며, 프롬프트는..."_
- **Case B (Fallback - 일반 조언):**
  - 도구 없이 작업을 수행하는 방법론, 팁, 노하우 제공.
  - 예: _"적합한 도구가 없습니다. 대신 발표 연습은 거울을 보거나 녹음해서 들어보는 방식을 추천합니다..."_

### 4.6 [Node-06] Reflection (메모리 저장)

- **기능:** 대화가 끝나면 백그라운드에서 실행.
- **프로세스:**
  1.  대화 내용에서 **사용자 선호도**(예: "무료 툴 선호", "코딩 관련 작업") 추출.
  2.  ChromaDB `user_profile` 컬렉션에 임베딩하여 저장(Upsert).

---

## 5. 데이터 및 알고리즘 전략

### 5.1 데이터 구축 전략

- **JSON (Static):** LangChain `JSONLoader` 사용. `data/tools.json` (도구 기본 정보).
- **PDF (Dynamic):** `PyPDFLoader` 사용. `data/updates/*.pdf` (SNS/뉴스 캡처).

### 5.2 임베딩 모델

- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **이유:** 한국어와 영어 쿼리를 모두 지원하며, 다국어 검색 성능이 우수함.

---

## 6. 개발 환경 및 기술 스택 (Tech Stack)

| 구분          | 기술 / 도구              | 비고                     |
| :------------ | :----------------------- | :----------------------- |
| **Language**  | Python 3.10+             |                          |
| **Framework** | LangChain, **LangGraph** | Graph 기반 에이전트 제어 |
| **LLM**       | OpenAI `gpt-4o-mini`     | 비용 효율성 및 성능 고려 |
| **Vector DB** | **ChromaDB**             | Local Persistent Storage |
| **Search**    | Google Custom Search API | 외부 정보 보완용         |
| **Backend**   | **FastAPI**              | 비동기 서버              |
| **Frontend**  | **Gradio**               | FastAPI Mount 방식 통합  |

---

## 7. 프로젝트 구조 (Project Scaffolding)

```text
AI-101/
├── data/
│   ├── tools_base.json      # 기본 도구 정보 (JSON)
│   └── updates/             # 최신 뉴스/SNS (PDF)
├── db/                      # ChromaDB 저장소 (Git 제외)
├── src/
│   ├── graph.py             # LangGraph 노드 및 엣지 정의
│   ├── tools.py             # Search, Calc, Time, RAG 툴 정의
│   ├── memory.py            # Reflection 및 DB 관리
│   └── prompts.py           # 상황별 프롬프트 관리
├── main.py                  # FastAPI 앱 + Gradio 마운트
├── requirements.txt         # 의존성 목록
└── .env                     # API Key 설정
```
