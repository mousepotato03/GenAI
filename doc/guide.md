## 1. LangGraph 상태 모델 (State Definition)

LangGraph의 **State(상태)**는 에이전트의 단기 메모리이자 모든 노드 간의 데이터 흐름을 정의하는 핵심 구조입니다. 프로젝트 요구사항에 맞춰 LangGraph `TypedDict`와 `Annotated list`를 사용하여 정의해야 합니다.

| 필드 이름                  | 타입                                  | 역할                                                                                                                                                                            | 소스                                                                                                                                                      |
| :------------------------- | :------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| **`messages`**             | `Annotated[List[dict], add_messages]` | **단기 메모리(Short Term Memory)**: 사용자 질문, LLM 응답, 그리고 **도구 실행 결과(Observation)**가 순서대로 누적됩니다. `add_messages` 리듀서 사용으로 자동 누적을 보장합니다. |                                                                                                                                                           |
| **`tool_result`**          | `str                                  | None`                                                                                                                                                                           | LLM이 요청한 도구 호출(Function Call) 정보(JSON 직렬화)가 저장됩니다. LLM 노드와 툴 노드 간의 ReAct 루프 분기(Conditional Edge)를 결정하는 데 사용됩니다. |     |
| **`is_complex_task`**      | `bool`                                | 사용자의 질문이 단순 Q&A인지, 서브태스크 분할이 필요한 복잡한 **작업 처리** 수준인지 판단한 결과입니다 (작업 1).                                                                |                                                                                                                                                           |
| **`sub_tasks`**            | `List[str]`                           | LLM이 분할한 실행 계획 목록입니다 (작업 2). (예: `["대본 작성", "영상 생성"]`)                                                                                                  |                                                                                                                                                           |
| **`tool_recommendations`** | `Dict[str, str]`                      | 각 `sub_task`에 대해 추천된 **최종 서비스/도구 정보**를 저장합니다 (작업 3).                                                                                                    | (프로젝트 정의)                                                                                                                                           |
| **`user_feedback`**        | `str                                  | None`                                                                                                                                                                           | Human Review 단계에서 사용자가 제공한 **수정/피드백 내용**을 임시로 저장하여 Re-planning에 활용합니다.                                                    |     |
| **`retrieved_docs`**       | `List[dict]`                          | RAG Tool (retrieve_docs) 또는 Google Search Tool의 검색 결과를 저장하여 LLM의 답변 생성 컨텍스트로 사용합니다.                                                                  |                                                                                                                                                           |
| **`final_guide`**          | `str                                  | None`                                                                                                                                                                           | 작업 5의 최종 작업 워크플로우 가이드 결과물입니다.                                                                                                        |     |

## 2. LangGraph 노드 구성 및 역할 (Node Configuration)

LangGraph는 ReAct 패턴을 구현하기 위해 최소한 `llm_agent`와 `tool_executor` 노드가 필요하며, 프로젝트의 복잡한 5단계 워크플로우를 처리하기 위해 추가적인 전용 노드들이 필요합니다.

| 노드 이름                      | 타입                  | 역할 (담당 작업)                                                                                                                                                             | 핵심 기능                       |
| :----------------------------- | :-------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------ |
| **`llm_router`** (Entry Point) | LLM Node              | **[작업 1]** 사용자 질문을 받아 단순 Q&A인지 복잡한 작업인지 판단하고 `is_complex_task` 플래그 설정.                                                                         | LLM 추론, State 업데이트.       |
| **`planning_node`**            | LLM Node              | **[작업 2]** 질문을 받아 최적의 서브태스크(`sub_tasks`) 목록으로 분리합니다. `user_feedback`이 있으면 Re-planning을 수행합니다.                                              | LLM 추론, State 업데이트.       |
| **`recommend_tool_node`**      | LLM Node (Tool Agent) | **[작업 3, 4]** 각 `sub_task`에 대해 RAG Tool (retrieve_docs)나 Search Tool, Calc/Time Tool을 호출하여 최적의 서비스 1개를 추천합니다. 이 노드는 ReAct 루프의 중심이 됩니다. | ReAct 루프, Tool Calling.       |
| **`tool_executor`**            | Tool Node             | LLM이 요청한 모든 도구 호출(`tool_result`)을 받아 실제 함수(RAG, Search, Calc, Time, Memory)를 실행하고 **Observation** 메시지를 반환합니다.                                 | 도구 실행 (Observation).        |
| **`guide_generation_node`**    | LLM Node              | **[작업 5]** 승인된 계획, 추천된 도구, 검색된 정보를 종합하여 **최종 워크플로우 가이드**를 생성합니다.                                                                       | 최종 답변 생성, State 업데이트. |
| **`reflection_node`**          | LLM/Tool Node         | **[작업 6]** 대화 완료 후, Memory Extractor Prompt를 사용하여 사용자 피드백/선호도를 추출하고 `write_memory` Tool을 호출하여 장기 메모리(Chroma DB)에 저장합니다.            | Memory Write, Reflection.       |

### 2.1 통합된 도구 (Tools)

프로젝트에 필요한 모든 도구는 `tool_executor` 노드에 등록되며, LLM에게는 **JSON 스키마** 형태로 전달됩니다.

1.  **RAG Tool (`retrieve_docs`):** JSON (기본 지식)과 PDF (최신 트렌드)를 **하이브리드**로 검색하는 핵심 도구입니다. Chroma DB와 다국어 임베더(`paraphrase-multilingual-MiniLM-L12-v2`)를 사용합니다.
2.  **Long Term Memory Tool (`read_memory`, `write_memory`):** Chroma DB Persistent를 활용하여 사용자의 선호도, 과거 피드백, 장기 목표 등을 벡터화하여 저장하고 검색합니다.
3.  **Utility Tools (`calculator`, `get_time`, `google_search`):** 가격 계산, 최신 정보 판단, 웹 검색 등 보조적인 역할에 사용됩니다.

## 3. 사용자 질문 처리 흐름 (LangGraph Workflow)

프로젝트의 다단계 작업은 **세 가지 주요 그래프 흐름**으로 구성되며, **Human-in-the-Loop**를 위해 LangGraph의 **Interrupt** 기능이 핵심적으로 사용됩니다.

### 단계 1: 작업 유형 분류 및 계획 수립

1.  **시작 (`START` → `llm_router`):** 사용자 질문이 그래프에 진입합니다. `llm_router`가 질문을 분석하여 **`is_complex_task`** 플래그를 결정합니다 (작업 1).
2.  **조건부 분기:**
    - **IF** `is_complex_task == False` (단순 Q&A): `guide_generation_node`로 바로 이동하여 답변을 생성하고 `END`로 종료합니다.
    - **IF** `is_complex_task == True` (복잡한 작업): `planning_node`로 이동합니다.
3.  **계획 수립 (`planning_node`):** LLM은 질문을 최적의 `sub_tasks` (계획) 목록으로 분할하여 State에 저장합니다.

### 단계 2: Human-in-the-Loop 검토 및 수정 (Interrupt)

이 단계는 **LangGraph의 `interrupt_before`** 기능을 사용하여 사용자의 승인/수정 피드백을 반영합니다.

1.  **중단 지점 설정:** LangGraph를 컴파일할 때, `planning_node` 다음에 실행될 `recommend_tool_node` 직전에 **Interrupt**를 설정합니다 (`interrupt_before=["recommend_tool_node"]`).
2.  **사용자 검토:** `planning_node` 실행 후 그래프가 멈춥니다.
    - Gradio UI는 State에서 `sub_tasks` 목록을 읽어 사용자에게 제시하고 승인을 요청합니다.
3.  **피드백 및 재개:**
    - **승인 시:** 클라이언트 코드는 `graph.invoke(None, config)`를 호출하여 **`recommend_tool_node`**로 실행을 재개합니다.
    - **수정 요청 시:** 사용자의 수정 내용은 `updated_msg` 형태로 변환됩니다. 클라이언트 코드는 `graph.update_state()` 함수를 사용하여 `sub_tasks` 또는 `user_feedback` 필드를 업데이트하고 (`as_node="planning_node"`), `planning_node`로 돌아가 Re-planning 루프를 수행하도록 설정하거나, `recommend_tool_node`로 넘어가기 전에 `planning_node`가 수정된 계획을 반영하도록 합니다.

### 단계 3: 도구 추천 및 ReAct 루프 (RAG/Tool Execution)

승인된 `sub_tasks`를 바탕으로 각 태스크에 맞는 서비스 추천을 수행합니다 (작업 3, 4). 이 과정은 **LLM 노드와 Tool 노드가 순환하는 ReAct 루프**를 따릅니다.

1.  **LLM 추론 (`recommend_tool_node`):** LLM은 현재 남은 `sub_task`를 확인하고, 이를 해결하기 위해 필요한 정보(기본 지식, 최신 정보, 가격 등)를 파악합니다.
2.  **Action (Tool Call):** LLM은 **RAG Tool** (`retrieve_docs`) 또는 **Search Tool**을 호출합니다. 이 호출 정보는 `tool_result`에 저장됩니다.
    - (예: `tool_calls: [{"name": "retrieve_docs", "arguments": {"query": "유튜브 쇼츠 제작 AI 도구 최신"}}]`)
3.  **분기 (`Conditional Edge`):**
    - **IF** `tool_result != None`: `tool_executor` 노드로 이동합니다.
4.  **Observation (`tool_executor`):** `tool_executor`는 `retrieve_docs` Tool을 실행하여 JSON 및 PDF 지식베이스에서 하이브리드 검색을 수행하고 **검색 결과**(`retrieved_docs`)를 얻습니다. 이 결과는 `role: "tool"` 메시지 형태로 `messages`에 추가됩니다.
5.  **LLM 재추론 (Loop Back):** `tool_executor`는 다시 **`recommend_tool_node`**로 연결됩니다. LLM은 검색 결과(Observation)를 보고 해당 도구의 적합성을 판단합니다 (유사도 임계값 체크 포함).
6.  **Tool Recommendation 저장:** LLM은 가장 적합한 서비스를 `tool_recommendations` 필드에 저장하고 다음 `sub_task`로 넘어갈지, 아니면 추가적인 정보(예: 계산기나 시간)가 필요한지 판단합니다 (작업 4).
7.  **Finalize Action:** 모든 `sub_task`에 대한 추천이 완료되면, LLM은 `tool_result`를 `None`으로 반환하여 Conditional Edge가 다음 단계로 이동하도록 합니다.

### 단계 4: 최종 가이드 생성 및 메모리 저장

1.  **최종 가이드 생성 (`guide_generation_node`):** 이 노드는 승인된 계획(`sub_tasks`)과 추천된 서비스(`tool_recommendations`)를 바탕으로, 검색 결과(`retrieved_docs`)를 활용하여 사용자가 작업을 실제로 수행할 수 있는 상세한 **워크플로우 가이드**를 작성하고 `final_guide`에 저장합니다 (작업 5).
2.  **종료 및 Reflection (`reflection_node`):**
    - **[작업 6]** 에이전트는 최종 답변(가이드)을 사용자에게 반환하기 전에, 이 노드를 실행하여 대화 내용을 분석합니다.
    - LLM(Memory Extractor Prompt 사용)을 호출하여 사용자 선호도나 피드백(예: "우리 지식 베이스에 없는 도구 요청")을 추출합니다.
    - 추출된 정보는 `write_memory` Tool을 통해 장기 메모리(Chroma DB Persistent)에 저장됩니다.
3.  **그래프 종료 (`END`):** 최종 답변을 반환하며 워크플로우를 종료합니다.

---

**비유:** 이 에이전트 시스템은 마치 **복잡한 프로젝트를 처리하는 컨설팅 팀**과 같습니다.

- **LangGraph**는 프로젝트 관리자(전체 흐름 관리)이며, **State**는 공유되는 프로젝트 문서입니다.
- **`planning_node`**는 기획자(작업 분해)이며, **Interrupt**는 고객(사용자)의 중간 승인 회의입니다.
- **`recommend_tool_node`**는 전문가 팀장(ReAct)이며, **RAG Tool**은 자체 구축된 지식 라이브러리(JSON, PDF)를 검색하는 전담 연구원입니다.
- 최종적으로 **`guide_generation_node`**가 보고서(최종 가이드)를 작성하고, **`reflection_node`**는 고객의 요구사항(선호도)을 다음 프로젝트를 위해 기록해 두는 역할을 합니다.
