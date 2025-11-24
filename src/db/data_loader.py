"""
데이터 로더
초기 AI 도구 데이터를 Vector DB에 로드
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from src.db.vector_store import VectorStore


def load_initial_data(vector_store: VectorStore, json_path: str = None) -> int:
    """
    초기 도구 데이터를 Vector DB에 로드

    Args:
        vector_store: VectorStore 인스턴스
        json_path: JSON 파일 경로 (기본값: data/ai_tools/initial_tools.json)

    Returns:
        로드된 도구 개수
    """
    if json_path is None:
        # 프로젝트 루트 디렉토리 기준 경로
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        json_path = project_root / "data" / "ai_tools" / "initial_tools.json"

    # JSON 파일 로드
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)

        print(f"JSON 파일 로드 성공: {len(tools)}개의 도구")

        # Vector DB에 추가
        vector_store.add_tools(tools)

        # 저장된 개수 확인
        count = vector_store.get_count()
        print(f"Vector DB에 {count}개의 도구 저장 완료")

        return len(tools)

    except FileNotFoundError:
        print(f"Error: JSON 파일을 찾을 수 없습니다: {json_path}")
        return 0
    except json.JSONDecodeError as e:
        print(f"Error: JSON 파싱 오류: {e}")
        return 0
    except Exception as e:
        print(f"Error: 데이터 로드 중 오류 발생: {e}")
        return 0


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("AI 101 - 초기 데이터 로더")
    print("=" * 50)

    # Vector Store 초기화
    print("\n1. Vector Store 초기화 중...")
    vector_store = VectorStore()

    # 기존 데이터 삭제 (선택사항)
    current_count = vector_store.get_count()
    if current_count > 0:
        print(f"   기존 데이터 {current_count}개 발견")
        response = input("   기존 데이터를 삭제하시겠습니까? (y/N): ")
        if response.lower() == 'y':
            vector_store.delete_all()
            print("   기존 데이터 삭제 완료")

    # 초기 데이터 로드
    print("\n2. 초기 데이터 로드 중...")
    loaded_count = load_initial_data(vector_store)

    if loaded_count > 0:
        print(f"\n✅ 성공: {loaded_count}개의 AI 도구 데이터 로드 완료!")

        # 테스트 검색
        print("\n3. 테스트 검색 수행...")
        test_queries = [
            "텍스트 생성",
            "이미지 만들기",
            "비디오 편집",
            "음성 합성"
        ]

        for query in test_queries:
            results, should_fallback = vector_store.search(query, n_results=3)
            print(f"\n   쿼리: '{query}'")
            print(f"   결과: {len(results)}개 (Fallback: {should_fallback})")
            if results:
                for i, result in enumerate(results, 1):
                    print(f"     {i}. {result['name']} (점수: {result['score']:.3f})")

    else:
        print("\n❌ 실패: 데이터 로드 실패")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
