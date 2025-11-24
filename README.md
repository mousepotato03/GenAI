# AI 101 - AI 도구 추천 에이전트

당신의 아이디어를 실행 가능한 단계로 분해하고, 최적의 AI 도구를 추천합니다!

## 📌 프로젝트 개요

AI 101은 사용자의 추상적인 아이디어를 분석하여:

1. 실행 가능한 서브태스크로 분해
2. 각 태스크에 맞는 검증된 AI 도구를 추천
3. 단계별 사용 가이드를 제공

하는 지능형 에이전트 시스템입니다.

## 🚀 빠른 시작

### 1. 환경 설정

````powershell
# 1. 가상환경 생성 및 활성화
python -m venv venv
.\venv\Scripts\Activate

# 2. 패키지 설치
pip install -r requirements.txt


```powershell
# AI 도구 데이터를 Vector DB에 로드
python src/db/data_loader.py
````

### 3. 서버 실행

#### 방법 1: 통합 실행 (권장)

```powershell
# API 서버 + UI 서버를 동시에 실행
.\start_all.ps1
```

#### 방법 2: 개별 실행

```powershell
# 터미널 1: API 서버
.\start_api.ps1

# 터미널 2: UI 서버
.\start_ui.ps1
```

### 4. 사용

브라우저에서 http://localhost:7860 접속

## 📖 사용 방법

1. **질문 입력**: 하고 싶은 작업을 자유롭게 입력하세요

   - 예: "유튜브 미스테리 쇼츠를 만들고 싶어"

2. **계획 확인**: AI가 생성한 단계별 계획을 확인하세요

3. **승인 또는 수정**: 계획을 승인하거나 수정 요청하세요

4. **가이드 받기**: 각 단계별 추천 도구와 사용 가이드를 받으세요!

## 🛠️ 기술 스택

- **Framework**: LangChain, LangGraph
- **LLM**: GPT-4o-mini
- **Vector DB**: ChromaDB
- **Embedding**: sentence-transformers
- **Search**: Google Custom Search API
- **Backend**: FastAPI
- **UI**: Gradio
- **Language**: Python 3.10+

## 📁 프로젝트 구조

```
GenAI/
├── src/
│   ├── agents/          # LangGraph 에이전트
│   │   ├── graph.py     # 워크플로우 통합
│   │   ├── state.py     # 상태 정의
│   │   └── nodes/       # 7개 노드
│   ├── db/              # 데이터베이스
│   │   ├── vector_store.py
│   │   └── data_loader.py
│   ├── tools/           # 검색/평가 도구
│   │   ├── search.py
│   │   └── evaluator.py
│   ├── api/             # FastAPI 서버
│   │   └── main.py
│   └── ui/              # Gradio UI
│       └── app.py
├── data/
│   ├── ai_tools/        # AI 도구 데이터
│   └── chroma_db/       # Vector DB 저장소
├── tests/               # 테스트
├── doc/                 # 문서
├── .env                 # 환경변수
└── requirements.txt     # 의존성
```

## 🔑 필수 API 키

### OpenAI API

1. https://platform.openai.com/api-keys 접속
2. API 키 생성
3. GPT-4o-mini 사용 가능한지 확인

### Google Custom Search API

1. https://console.cloud.google.com/ 접속
2. 프로젝트 생성
3. Custom Search API 활성화
4. API 키 생성
5. https://programmablesearchengine.google.com/ 에서 검색 엔진 생성

## 📊 워크플로우

```
사용자 질문
    ↓
1. Plan (계획 생성)
    ↓
2. Review (승인 대기) ← Human-in-the-loop
    ↓ (승인)
3. Research (도구 검색) - 병렬 처리
    ↓
4. Evaluate (평판 조회)
    ↓
5. Guide (가이드 생성)
    ↓
6. Synthesize (응답 통합)
    ↓
7. Memory (선호도 학습)
    ↓
최종 응답
```

## 🎯 주요 기능

- ✅ **자동 태스크 분해**: 복잡한 목표를 실행 가능한 단계로 분해
- ✅ **AI 도구 추천**: RAG + Web Search로 최적의 도구 검색
- ✅ **평판 기반 평가**: 웹 검색으로 도구의 평판 조회
- ✅ **사용 가이드 제공**: 초보자도 따라할 수 있는 단계별 가이드
- ✅ **Fallback 모드**: 특화 도구가 없으면 범용 LLM 프롬프트 제공
- ✅ **선호도 학습**: 사용자 패턴을 학습하여 개인화된 추천

## 📚 예제 질문

- "유튜브 미스테리 쇼츠를 만들고 싶어. 시나리오부터 영상, 더빙까지 전부"
- "블로그 글을 쓰고 싶은데, 주제는 AI 활용법이야. 썸네일도 만들어줘"
- "회사 발표용 PPT를 만들고 싶어. 디자인도 예쁘게"
- "Python 웹 크롤러를 만들고 싶어. 코드 작성부터 디버깅까지"

## 📝 API 문서

서버 실행 후 http://localhost:8000/docs 접속

주요 엔드포인트:

- `POST /chat`: 사용자 질문 처리
- `POST /approve`: 계획 승인/거절
- `GET /status/{session_id}`: 세션 상태 조회
