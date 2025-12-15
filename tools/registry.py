"""
Tool Registry - 도구 등록 및 실행
"""
from typing import List, Any

from tools.search import retrieve_docs, google_search_tool
from tools.memory_tools import read_memory, write_memory
from tools.calculator import calculate_subscription_cost, calculate_math
from tools.time_tools import check_tool_freshness, get_current_time


def get_all_tools() -> List:
    """LangChain 도구 리스트 반환 (LLM 바인딩용)"""
    return [
        retrieve_docs,
        read_memory,
        write_memory,
        google_search_tool,
        calculate_subscription_cost,
        calculate_math,
        check_tool_freshness,
        get_current_time
    ]


def execute_tool(tool_name: str, args: dict) -> Any:
    """
    도구 이름으로 실행

    Args:
        tool_name: 도구 이름
        args: 도구 인자 딕셔너리

    Returns:
        도구 실행 결과
    """
    tools_map = {
        "retrieve_docs": retrieve_docs,
        "read_memory": read_memory,
        "write_memory": write_memory,
        "google_search_tool": google_search_tool,
        "calculate_subscription_cost": calculate_subscription_cost,
        "calculate_math": calculate_math,
        "check_tool_freshness": check_tool_freshness,
        "get_current_time": get_current_time
    }

    tool_func = tools_map.get(tool_name)
    if tool_func:
        return tool_func.invoke(args)
    raise ValueError(f"알 수 없는 도구: {tool_name}")
