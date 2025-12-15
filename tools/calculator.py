"""
Calculator Tools - 비용 계산 및 수학 계산 도구
"""
from typing import List, Dict

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
def calculate_math(expression: str) -> str:
    """
    간단한 수학 계산을 수행합니다.

    Args:
        expression: 계산할 수식 (예: "22 * 34", "100 + 50", "1000 / 4")

    Returns:
        계산 결과

    Examples:
        - "22 * 34" -> "748"
        - "100 + 50 - 20" -> "130"
    """
    try:
        # 안전한 계산을 위해 허용된 문자만 사용
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "오류: 허용되지 않는 문자가 포함되어 있습니다. 숫자와 +, -, *, /, (, )만 사용 가능합니다."

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"
