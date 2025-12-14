"""
Tool Executor Node - 도구 실행
"""
import json
from typing import Dict

from langchain_core.messages import ToolMessage

from agent.state import AgentState
from tools.registry import execute_tool


def tool_executor_node(state: AgentState) -> Dict:
    """
    도구 실행 노드

    recommend_tool_node가 요청한 Tool Call을 실제로 실행합니다.
    """
    print("[Node] tool_executor 실행")

    tool_result = state.get("tool_result")
    retrieved_docs = state.get("retrieved_docs", [])

    if not tool_result:
        print("  - tool_result 없음, 스킵")
        return {}

    try:
        tool_call = json.loads(tool_result)
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        tool_id = tool_call.get("id", "")

        print(f"  - 실행 도구: {tool_name}")
        print(f"  - 인자: {tool_args}")

        # 도구 실행
        result = execute_tool(tool_name, tool_args)
        observation = str(result)

        print(f"  - 결과 길이: {len(observation)} chars")

        # retrieved_docs 업데이트 (검색 결과인 경우)
        task_completed = False  # should_fallback: False이면 태스크 완료 처리

        if tool_name in ["retrieve_docs", "google_search_tool"]:
            try:
                result_data = json.loads(observation)

                # retrieve_docs: 2단계 점수 합산 방식 (recommended_tool + candidates)
                if "recommended_tool" in result_data:
                    recommended = result_data.get("recommended_tool")
                    candidates = result_data.get("candidates", [])

                    if recommended:
                        retrieved_docs.append(recommended)
                        print(f"  - retrieved_docs에 추천 도구 1개 추가")

                        # 점수 출력
                        scores = recommended.get("scores", {})
                        print(f"  - [추천 도구] {recommended.get('name')}")
                        print(f"    • json_score: {scores.get('json_score', 0):.3f}")
                        print(f"    • pdf_score: {scores.get('pdf_score', 0):.3f}")
                        print(f"    • final_score: {scores.get('final_score', 0):.3f}")

                        # 후보군 출력
                        if candidates:
                            print(f"  - [후보군]")
                            for c in candidates[:3]:
                                print(f"    • {c.get('name')}: {c.get('final_score', 0):.3f}")

                    # should_fallback 상태 출력 및 태스크 완료 판단
                    should_fallback = result_data.get("should_fallback", False)
                    print(f"  - should_fallback: {should_fallback} (threshold 미달 여부)")

                    if not should_fallback:
                        task_completed = True  # 충분한 결과 → 태스크 완료

                # google_search_tool: 기존 방식 (results 리스트)
                else:
                    docs = result_data.get("results", [])
                    if isinstance(docs, list):
                        retrieved_docs.extend(docs)
                        print(f"  - retrieved_docs에 {len(docs)}개 추가")

                        # 유사도 점수 출력
                        print(f"  - [유사도 점수]")
                        for doc in docs:
                            score = doc.get("score", 0)
                            name = doc.get("name", doc.get("content", "")[:30])
                            print(f"    • {name}: {score:.4f}")

                        # should_fallback 상태 출력
                        should_fallback = result_data.get("should_fallback", False)
                        print(f"  - should_fallback: {should_fallback} (threshold 미달 여부)")
            except:
                pass

        # ToolMessage로 Observation 반환
        result = {
            "retrieved_docs": retrieved_docs,
            "messages": [ToolMessage(content=observation, tool_call_id=tool_id)],
            "tool_result": None  # 다음 루프를 위해 초기화
        }

        # should_fallback: False이면 충분한 결과 → 다음 태스크로 이동
        if task_completed:
            current_idx = state.get("current_task_idx", 0)
            result["current_task_idx"] = current_idx + 1
            result["tool_call_count"] = 0  # 다음 태스크를 위해 초기화
            print(f"  - 충분한 결과 획득, 다음 태스크로 이동 (idx: {current_idx} → {current_idx + 1})")

        return result

    except Exception as e:
        print(f"  - 도구 실행 오류: {e}")
        return {
            "messages": [ToolMessage(content=f"도구 실행 오류: {str(e)}", tool_call_id="error")],
            "tool_result": None,
            "error": str(e)
        }
