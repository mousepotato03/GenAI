"""
Calculator Tools - 비용 계산 및 시간 관련 도구
"""
from typing import List, Dict
from datetime import datetime

from langchain_core.tools import tool


@tool
def calculate_subscription_cost(tool_names: List[str], tool_prices: List[float]) -> str:
    """
    선택된 AI 도구들의 월간 구독료를 합산합니다.

    Args:
        tool_names: 도구 이름 리스트
        tool_prices: 각 도구의 월간 가격 리스트 (USD)

    Returns:
        구독료 합산 결과 문자열
    """
    if len(tool_names) != len(tool_prices):
        return "오류: 도구 이름과 가격 리스트의 길이가 일치하지 않습니다."

    total = sum(tool_prices)
    yearly = total * 12

    result_lines = ["## 📊 월간 구독료 계산 결과\n"]

    for name, price in zip(tool_names, tool_prices):
        if price == 0:
            result_lines.append(f"- **{name}**: 무료")
        else:
            result_lines.append(f"- **{name}**: ${price:.2f}/월")

    result_lines.append(f"\n### 💰 총 비용")
    result_lines.append(f"- **월간**: ${total:.2f}")
    result_lines.append(f"- **연간**: ${yearly:.2f}")

    if total > 50:
        result_lines.append(f"\n> ⚠️ 월 $50 이상의 비용이 예상됩니다. 무료 대안을 고려해보세요.")

    return "\n".join(result_lines)


def calculate_tools_cost(tools: List[Dict]) -> Dict:
    """
    도구 리스트에서 비용 정보 수집 (내부 함수)

    Args:
        tools: 도구 정보 딕셔너리 리스트

    Returns:
        비용 정보 딕셔너리
    """
    breakdown = []

    for tool in tools:
        name = tool.get('name', 'Unknown')
        pricing_model = tool.get('pricing_model', '')
        pricing_notes = tool.get('pricing_notes', '')

        breakdown.append({
            "name": name,
            "pricing_model": pricing_model,
            "pricing_notes": pricing_notes
        })

    return {
        "breakdown": breakdown,
        "note": "정확한 비용은 각 서비스 공식 사이트에서 확인하세요."
    }


@tool
def check_tool_freshness(tool_name: str, updated_date: str) -> str:
    """
    AI 도구 정보의 최신성을 확인합니다.

    Args:
        tool_name: 도구 이름
        updated_date: 도구 정보 업데이트 날짜 (YYYY-MM-DD 형식)

    Returns:
        최신 여부 판단 결과
    """
    try:
        today = datetime.now()
        update_dt = datetime.strptime(updated_date, "%Y-%m-%d")
        days_old = (today - update_dt).days

        if days_old <= 30:
            return f"✅ '{tool_name}' 정보는 최신입니다. ({days_old}일 전 업데이트)"
        elif days_old <= 90:
            return f"⚠️ '{tool_name}' 정보가 다소 오래되었습니다. ({days_old}일 전 업데이트) 공식 사이트에서 최신 정보를 확인하세요."
        else:
            return f"❌ '{tool_name}' 정보가 오래되었습니다. ({days_old}일 전 업데이트) 반드시 공식 사이트에서 확인하세요."

    except ValueError:
        return f"⚠️ '{tool_name}'의 업데이트 날짜 형식이 올바르지 않습니다."


@tool
def get_current_time() -> str:
    """
    현재 날짜와 시간을 반환합니다.
    도구 정보의 최신성을 확인하거나 시간 관련 정보가 필요할 때 사용하세요.

    Returns:
        현재 날짜/시간 문자열 (YYYY-MM-DD HH:MM:SS 형식)
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def check_freshness_simple(updated_date: str) -> Dict:
    """
    최신 여부를 간단히 확인 (내부 함수)

    Returns:
        {"is_fresh": bool, "days_old": int, "message": str}
    """
    try:
        today = datetime.now()
        update_dt = datetime.strptime(updated_date, "%Y-%m-%d")
        days_old = (today - update_dt).days

        return {
            "is_fresh": days_old <= 30,
            "days_old": days_old,
            "message": "최신" if days_old <= 30 else "확인 필요"
        }
    except:
        return {
            "is_fresh": False,
            "days_old": -1,
            "message": "날짜 형식 오류"
        }
