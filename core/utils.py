"""
Core Utils - 공통 유틸리티 함수
"""
import re
import json


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
    
    # 패턴 3: 그냥 JSON으로 시작하는 경우 - 첫 번째 완전한 JSON만 추출
    if text.startswith('{') or text.startswith('['):
        return _extract_first_valid_json(text)
    
    # 패턴 4: 텍스트 중간에 JSON이 있는 경우 - 첫 번째 완전한 JSON만 추출
    # 객체 찾기
    obj_match = re.search(r'\{', text)
    # 배열 찾기
    arr_match = re.search(r'\[', text)
    
    # 더 먼저 나오는 것부터 시도
    if obj_match and (not arr_match or obj_match.start() < arr_match.start()):
        return _extract_first_valid_json(text[obj_match.start():])
    elif arr_match:
        return _extract_first_valid_json(text[arr_match.start():])
    
    return text


def _extract_first_valid_json(text: str) -> str:
    """텍스트에서 첫 번째 유효한 JSON 객체/배열 추출"""
    # 중괄호나 대괄호로 시작하는지 확인
    if not (text.startswith('{') or text.startswith('[')):
        return text
    
    # 브레이스 카운팅으로 첫 번째 완전한 JSON 찾기
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            
        # 완전한 JSON 객체/배열을 찾았을 때
        if brace_count == 0 and bracket_count == 0 and i > 0:
            candidate = text[:i+1]
            # 실제로 파싱 가능한지 검증
            try:
                json.loads(candidate)
                return candidate
            except:
                continue
    
    # 완전한 JSON을 못 찾았으면 원본 반환
    return text
