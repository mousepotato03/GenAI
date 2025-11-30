## 1. 프로젝트를 위한 전체적인 흐름 (LangGraph/ReAct 순환 구조)

사용자의 질문이 들어왔을 때부터 최종 답변을 반환하기까지의 과정은 **ReAct 루프**를 따르며, 이는 LangGraph의 **노드(Node)**와 **조건부 에지(Conditional Edge)**를 통해 구현됩니다.

### A. 핵심 LangGraph 구성 요소 정의

에이전트의 상태(State)와 논리적인 흐름을 정의하는 것이 가장 먼저 수행되어야 합니다.

| 구성 요소                  | 역할 및 설명                                                                                                                                                                                            | 소스 인용 |
| :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------- |
| **State (상태 모델)**      | 그래프 전체가 공유하는 데이터 저장소입니다. `messages`와 `tool_result`가 핵심이며, `TypedDict`와 `Annotated list`를 사용하여 정의해야 합니다.                                                           |           |
| **Messages (단기 메모리)** | 지금까지의 대화 히스토리와 툴 호출 기록(Function Call history, Observation)을 저장하며, **단기 메모리** 역할을 합니다. `Annotated[List[dict], add_messages]`를 사용하여 자동으로 누적되도록 관리합니다. |           |
| **Tool Result**            | LLM이 도구 호출(Tool Call)을 결정했을 경우, 해당 호출 정보(함수 이름, 인수)가 저장되는 필드입니다. 이 값이 `None`인지 아닌지에 따라 다음 단계(도구 실행 또는 종료)가 결정됩니다.                        |           |
| **Nodes**                  | `State`를 입력받아 새로운 `State`를 반환하는 함수입니다. 최소한 **LLM 노드**와 **Tool 노드**가 필요합니다.                                                                                              |           |
| **Conditional Edge**       | LLM 노드의 출력(tool_result)을 기반으로 그래프의 흐름을 **분기**시키는 지능적인 기능입니다. ReAct 루프를 구현하는 핵심입니다.                                                                           |           |

### B. 사용자 질문 처리 흐름 (ReAct Loop in LangGraph)

1.  **시작 및 입력:**

    - 사용자가 Gradio UI를 통해 질문을 입력합니다.
    - LangGraph는 초기 `State` (메시지: 사용자 질문, `tool_result`: `None`)를 가지고 시작합니다.
    - **시작 지점 (Entry Point):** `llm` 노드로 설정됩니다.

2.  **LLM 1차 호출 (Thought / Action 추론) - `llm_node` 실행:**

    - `llm_node`는 현재 `messages` (대화 기록)와 정의된 모든 `tools` (구글 검색, 계산기, RAG, 메모리 등)를 포함하여 **`gpt-4o-mini`**를 호출합니다.
    - LLM은 질문을 분석하여 **'생각(Thought)'**을 하고, 다음 **'행동(Action)'**을 결정합니다.
      - **경우 1: 도구 호출이 필요한 경우 (Action):** LLM은 도구 호출 객체(`tool_calls`)를 반환합니다. `llm_node`는 이 호출 정보를 `tool_result`에 JSON 형태로 저장하고, LLM의 응답(tool_calls)을 `messages`에 추가한 새로운 `State`를 반환합니다.
      - **경우 2: 최종 답변이 가능한 경우 (Final):** LLM은 최종 답변 텍스트를 반환하고, `tool_calls`는 `None`입니다. `llm_node`는 LLM의 답변을 `messages`에 추가하고 `tool_result`를 `None`으로 설정한 `State`를 반환합니다.

3.  **흐름 분기 (Conditional Edge) - `route` 함수 실행:**

    - LangGraph는 `llm_node`가 반환한 `State`를 검사합니다.
    - **`tool_result`가 `None`이 아니면** (`tool` 노드로 이동).
    - **`tool_result`가 `None`이면** (`END`로 이동, 최종 답변 반환).

4.  **도구 실행 (Action → Observation) - `tool_node` 실행:**

    - `tool_node`는 `State`에서 `tool_result`를 읽어들여 실제 Python 함수/클래스를 호출합니다.
      - 예: `tool_result`가 `calculator` 호출을 지시하면, `tool_node`는 `calculator()` 함수를 실행합니다.
    - 도구 실행 결과는 **'관찰(Observation)'**이 됩니다.
    - `tool_node`는 이 Observation을 `role: "tool"` 메시지 형식으로 변환하고, 이를 기존 `messages`에 추가합니다.
    - `tool_node`는 `tool_result`를 다시 `None`으로 설정한 `State`를 반환하며, **다시 `llm_node`로 돌아가는 에지**를 따라 루프를 반복합니다.

5.  **LLM 2차 호출 (최종 답변 생성):**

    - `llm_node`가 관찰 결과(`role: "tool"` 메시지)를 포함한 새로운 `messages` 리스트를 가지고 다시 호출됩니다.
    - LLM은 이 관찰 결과(RAG 문서, 계산 값, 메모리 내용 등)를 활용하여 최종 사용자 친화적인 답변을 생성합니다.
    - `tool_calls`가 `None`인 응답을 반환하고, 그래프는 `END`로 종료됩니다.

6.  **(선택) Reflection/자동 메모리 저장:**
    - 최종 답변이 생성된 후, **Reflection** 기능을 구현하기 위해 별도의 노드 또는 최종 처리 단계에서 **자동 메모리 저장** 로직을 실행할 수 있습니다.
    - 이 단계에서는 LLM(Memory Extractor Prompt 사용)을 호출하여 이번 턴의 대화 내용 중 장기 메모리(Chroma DB)에 저장할 가치가 있는 내용(`profile`, `episodic`, `knowledge`)이 있는지 판단하고 `write_memory` 도구를 호출하거나 직접 저장합니다.

---

## 2. LLM, 툴, RAG, 메모리 구현 상세

프로젝트 요구사항에 따라 구현해야 할 구체적인 요소들은 다음과 같습니다.

### ① LLM 활용

- **모델:** `gpt-4o-mini`를 기본 LLM으로 사용합니다.
- **API:** OpenAI 기본 API를 사용하여 `client.chat.completions.create` 함수를 호출하며, 이때 `tools` 파라미터에 사용할 도구들의 JSON 스키마를 전달해야 합니다.

### ② Tool – Tool Calling 구현

모든 도구는 LLM에게 JSON 스키마 형태로 정의되어야 하며, LLM이 이를 보고 호출을 결정합니다.

| 도구 이름                           | 역할 및 구현 내용                                                                                                       | 소스 인용 |
| :---------------------------------- | :---------------------------------------------------------------------------------------------------------------------- | :-------- |
| **계산기 (Calculator)**             | 수학적 표현식(`expr`)을 입력받아 결과를 반환합니다. `eval()` 함수를 이용한 간단한 구현 예시가 소스에 제시되어 있습니다. |           |
| **구글 검색 API**                   | 웹 검색이 필요한 질의에 대해 외부 정보를 검색하여 결과를 반환합니다.                                                    |           |
| **시간**                            | 현재 시간을 문자열로 반환합니다. `get_time` 함수 예시가 소스에 제시되어 있으며, 타임존 인수를 가질 수 있습니다.         |           |
| **RAG Tool (retrieve_docs)**        | PDF 문서 검색을 담당하는 핵심 도구입니다.                                                                               |           |
| **Memory Tool (read/write_memory)** | 장기 메모리 접근을 위한 도구입니다.                                                                                     |           |

### ③ RAG – Tool Calling 구현 (지식/문서 메모리)

RAG(검색 증강 생성)는 에이전트가 외부 문서를 검색하여 정보를 프롬프트에 주입하는 구조입니다. 프로젝트에서는 디렉토리의 PDF 파일을 색인하고 검색하는 도구를 구현해야 합니다.

1.  **전처리 (색인 구축):**

    - PDF Reader를 사용하여 문서를 읽어옵니다.
    - 텍스트 분리기(Sentence Splitter)를 사용하여 문서를 검색하기 쉬운 작은 청크로 분할합니다.
    - **한영 multi-ligual embedder** (예: `"paraphrase-multilingual-MiniLM-L12-v2"`)를 사용하여 각 청크를 고차원 벡터로 변환합니다.
    - 이 벡터와 원본 텍스트를 **Chroma DB** (Persistent DB)에 저장하고 HNSW 색인 기법을 사용하여 검색 효율성을 높입니다.

2.  **RAG 실행 (Tool 호출):**
    - LLM이 사용자 질문을 보고 RAG가 필요하다고 판단하면, `retrieve_docs` (RAG Tool)를 호출합니다.
    - `tool_node`는 쿼리를 임베딩하여 Chroma DB에서 **의미적 유사성**이 높은 문서를 검색합니다.
    - 검색된 문서(컨텍스트)는 `Observation`으로 LLM에 다시 전달되며, LLM은 이 컨텍스트를 바탕으로 최종 답변을 생성합니다.

### ④ Memory – Tool Calling 구현

프로젝트는 단기 메모리와 장기 메모리를 모두 구현해야 합니다.

1.  **단기 메모리 (Short Term Memory):**

    - **LangGraph State**의 `messages` 필드 자체가 Short Term 메모리 역할을 합니다.
    - `messages`는 `Annotated list`를 사용하여 이전 대화의 흐름과 툴 호출 결과(Observation)를 자동으로 누적하여 관리합니다.

2.  **장기 메모리 (Long Term Memory):**

    - **Chroma DB Persistent**를 사용하여 과거 대화 기록이나 사용자 프로필 정보를 벡터화하여 저장합니다.
    - **`read_memory`** Tool을 호출하여 사용자가 과거에 언급한 내용이나 중요한 에피소드(경험) 메모리를 검색합니다. 검색 원리는 RAG와 유사하게 의미적 유사성을 기반으로 합니다.
    - **`write_memory`** Tool을 사용하여 새로운 정보를 장기적으로 저장합니다.

3.  **Reflection (자동 메모리 저장):**
    - 자동 메모리 저장을 구현하여 한 턴이 끝난 후, LLM이 대화 내용을 분석하여 저장할 가치가 있는지 판단하고 `write_memory` 도구를 호출하거나 직접 DB에 기록하는 파이프라인이 필요합니다.
    - 이 과정에서 **Memory Extractor Prompt**를 사용하여 메모리 타입(프로필, 에피소드, 지식)과 중요도를 구조화된 JSON 형태로 추출할 수 있습니다.

### ⑤ Graph 엔진 – LangGraph 고급 기능 활용

LangGraph의 주요 특징은 Agent의 상태를 저장하고 흐름을 제어하는 것입니다.

- **State:** 위에서 설명한 `Annotated list`를 활용하여 `messages` 히스토리 누적을 자동화합니다.
- **Interrupt 기능:** 특정 노드(예: 최종 답변 직전)에서 실행을 멈추고 사람의 검토나 입력을 받은 뒤, `update_state()` 함수를 사용하여 상태를 업데이트하고 다시 실행을 재개(Resume)하는 **Human-In-The-Loop** 기능을 구현해야 합니다.
- **Stream 기능:** 그래프 실행 과정을 단계별로 모니터링하여, 각 노드의 완료 시점마다 결과를 실시간으로 사용자에게 보여주는 기능을 구현합니다. `graph.stream(inputs)` 메서드를 사용할 수 있습니다.

---

## 3. LangGraph 적용을 통한 ReAct 루프 완성

전반적으로, 이 프로젝트는 단일 Python 스크립트 내에서 ReAct 루프를 구현하는 것이 아니라, LangGraph라는 오케스트레이션 프레임워크를 사용하여 LLM 호출, 툴 실행, 분기 로직(ReAct)을 명확하게 **모듈화**하고 **그래프화**하는 데 있습니다.

LangGraph를 사용하면, 기본 Tool Calling 워크플로우를 루프(Loop) 구조로 쉽게 변환할 수 있습니다.

**LangGraph가 ReAct를 구현하는 방식:**

| ReAct 단계             | LangGraph 요소                          | 데이터 흐름 (State 업데이트)                                                                     |
| :--------------------- | :-------------------------------------- | :----------------------------------------------------------------------------------------------- |
| **Thought** (추론)     | LLM Node 내부 (LLM의 비노출 영역)       | LLM이 `messages`를 읽고 행동을 결정.                                                             |
| **Action** (행동)      | LLM Node 출력 (`tool_result` != `None`) | LLM이 `tool_calls`를 반환하고, 이것이 `tool_result`에 저장됨.                                    |
| **Observation** (관찰) | Tool Node 실행                          | `tool_node`가 `tool_result`의 도구를 실행하고, 결과를 `role: "tool"` 메시지로 `messages`에 추가. |
| **Repeat / Final**     | Conditional Edge (`route` 함수)         | `tool_result`가 `None`이 아니면 `tool` → `llm` 루프를 반복하고, `None`이면 `END`.                |

LangGraph는 마치 **컨베이어 벨트 위의 공장 관리자**와 같습니다. 사용자 질문이라는 원료가 들어오면, LLM 노드라는 작업자에게 상태(State)를 전달하고, 작업자가 도구 호출이라는 청사진을 내놓으면, LangGraph는 컨베이어 벨트를 Tool 노드로 돌려 실제 작업을 실행(Observation)하게 한 다음, 다시 LLM 작업자에게 결과와 함께 상태를 돌려보내 최종 제품(Final Answer)이 나올 때까지 이 과정을 반복하게 합니다. 이 모든 과정에서 `State`가 변경될 때마다 체크포인트를 저장(persistence)할 수 있어, 에이전트가 이전의 기억을 잃지 않게 됩니다.
