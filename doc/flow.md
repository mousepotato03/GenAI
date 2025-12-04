● 사용자 질문 처리 흐름

  1. 요청 수신

  사용자 입력
      ↓
  main.py → app/api/routes.py (또는 app/ui/gradio_app.py)

  Gradio UI 경로:
  - app/ui/gradio_app.py → process_message() 함수가 메시지 수신

  REST API 경로:
  - app/api/routes.py → start_chat() 엔드포인트가 요청 수신

  ---
  2. 그래프 초기화 및 실행

  # app/api/routes.py:67 또는 app/ui/gradio_app.py:38
  graph = create_agent_graph()
  initial_state = create_initial_state(message, user_id)

  관련 파일:
  - agent/graph.py → create_agent_graph(), create_initial_state()
  - agent/state.py → AgentState TypedDict 정의

  ---
  3. 노드 실행 순서

  [1] llm_router_node (질문 유형 분류)
           ↓
      ┌────┴────┐
      ↓         ↓
   단순 Q&A   복잡한 작업
      ↓         ↓
      │    [2] planning_node (서브태스크 분해)
      │         ↓
      │    [INTERRUPT] ← 사용자 승인 대기
      │         ↓
      │    [3] recommend_tool_node (도구 추천)
      │         ↓ ↺
      │    [4] tool_executor_node (도구 실행)
      │         ↓
      └────→[5] guide_generation_node (최종 가이드)
                ↓
           [6] reflection_node (메모리 저장)
                ↓
              END

  ---
  4. 각 노드별 파일 및 역할

  | 순서  | 노드                    | 파일                        | 역할                |
  |-----|-----------------------|---------------------------|-------------------|
  | 1   | llm_router_node       | agent/nodes/router.py     | 단순 Q&A / 복잡 작업 분류 |
  | 2   | planning_node         | agent/nodes/planning.py   | 2-5개 서브태스크로 분해    |
  | 3   | recommend_tool_node   | agent/nodes/recommend.py  | ReAct 패턴으로 도구 추천  |
  | 4   | tool_executor_node    | agent/nodes/executor.py   | Tool Call 실행      |
  | 5   | guide_generation_node | agent/nodes/guide.py      | 최종 가이드 생성         |
  | 6   | reflection_node       | agent/nodes/reflection.py | 사용자 선호도 저장        |

  ---
  5. 라우팅 결정

  파일: agent/routing.py

  # llm_router 후 분기
  def route_after_llm_router(state):
      if state["is_complex_task"]:
          return "planning_node"      # 복잡한 작업
      else:
          return "guide_generation_node"  # 단순 Q&A

  # ReAct 루프 분기
  def route_after_recommend(state):
      if state["tool_result"]:
          return "tool_executor"      # 도구 실행
      elif current_idx < len(sub_tasks):
          return "recommend_tool_node"  # 다음 태스크
      else:
          return "guide_generation_node"  # 완료

  ---
  6. Human-in-the-Loop (HITL)

  파일: agent/hitl.py

  planning_node 완료 후
           ↓
      [INTERRUPT] ← interrupt_before=["recommend_tool_node"]
           ↓
  사용자 응답 ("좋아 진행해" / "2번 빼줘" / "취소")
           ↓
  handle_human_feedback() → analyze_user_intent()
           ↓
      ┌────┴────┬────┐
      ↓         ↓    ↓
    approve   modify  cancel
      ↓         ↓    ↓
    진행    계획수정  종료

  ---
  7. 도구 호출 흐름

  ReAct 루프 (recommend_tool_node ↔ tool_executor_node):

  recommend_tool_node
      ↓
  LLM이 Tool Call 생성
      ↓ (tool_result에 저장)
  tool_executor_node
      ↓
  tools/registry.py → execute_tool()
      ↓
  tools/search.py (retrieve_docs, google_search_tool)
  tools/memory_tools.py (read_memory, write_memory)
  tools/calculator.py (calculate_subscription_cost)
      ↓
  결과 → ToolMessage로 반환
      ↓
  recommend_tool_node (다시 루프)

  ---
  8. 프롬프트 사용

  각 노드는 prompts/ 폴더의 해당 파일에서 프롬프트를 가져옴:

  | 노드         | 프롬프트 파일               |
  |------------|-----------------------|
  | router     | prompts/router.py     |
  | planning   | prompts/planning.py   |
  | recommend  | prompts/recommend.py  |
  | guide      | prompts/guide.py      |
  | reflection | prompts/reflection.py |
  | HITL       | prompts/intent.py     |

  ---
  9. 전체 호출 스택 예시 (복잡한 작업)

  1. app/ui/gradio_app.py:process_message()
  2.   → agent/graph.py:create_agent_graph()
  3.   → agent/graph.py:create_initial_state()
  4.   → graph.stream(initial_state, config)
  5.     → agent/nodes/router.py:llm_router_node()
  6.       → core/llm.py:get_llm()
  7.       → prompts/router.py (프롬프트)
  8.     → agent/routing.py:route_after_llm_router() → "planning_node"
  9.     → agent/nodes/planning.py:planning_node()
  10.      → prompts/planning.py (프롬프트)
  11.    [INTERRUPT] ← 사용자 승인 대기
  12.  → agent/hitl.py:handle_human_feedback()
  13.    → prompts/intent.py (프롬프트)
  14.  → graph.stream(None, config) ← 재개
  15.    → agent/nodes/recommend.py:recommend_tool_node()
  16.      → tools/registry.py:get_all_tools()
  17.      → prompts/recommend.py (프롬프트)
  18.    → agent/nodes/executor.py:tool_executor_node()
  19.      → tools/search.py:retrieve_docs()
  20.      → core/memory.py:hybrid_search()
  21.    → agent/nodes/guide.py:guide_generation_node()
  22.      → prompts/guide.py (프롬프트)
  23.    → agent/nodes/reflection.py:reflection_node()
  24.      → core/memory.py:save_user_profile()
  25.    → END
  26. → 응답 반환
