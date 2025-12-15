"""
Core Utils - 공통 유틸리티 함수
"""
import re


def extract_json(text: str) -> str:
    """LLM 응답에서 JSON을 안전하게 추출"""
    text = text.strip()
    
    # 패턴 1: ```json ... ```
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 패턴 2: ``` ... ```
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 패턴 3: 그냥 JSON으로 시작
    if text.startswith('{'):
        return text
    
    # 패턴 4: 텍스트 중간에 JSON이 있는 경우
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    
    return text
