"""
평판 조회 도구 (Reputation Evaluator)

Web Search를 통해 AI 도구의 평판을 조회하고 점수를 산정합니다.
"""

import os
import re
from typing import Optional
from googleapiclient.discovery import build


class ReputationEvaluator:
    """AI 도구의 평판을 조회하고 점수를 산정하는 클래스"""

    def __init__(self):
        """
        환경변수에서 Google API 키를 로드합니다.
        """
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

        if not self.google_api_key or not self.search_engine_id:
            print("[ReputationEvaluator] 경고: Google API 설정이 없습니다. 평판 조회를 건너뜁니다.")
            self.enabled = False
        else:
            self.google_service = build("customsearch", "v1", developerKey=self.google_api_key)
            self.enabled = True

        # 긍정/부정 키워드
        self.positive_keywords = [
            "best", "great", "excellent", "amazing", "love", "recommend",
            "powerful", "useful", "helpful", "efficient", "easy",
            "좋", "최고", "훌륭", "추천", "유용", "편리"
        ]

        self.negative_keywords = [
            "bad", "worst", "terrible", "awful", "hate", "disappointed",
            "useless", "difficult", "expensive", "slow", "buggy",
            "나쁘", "최악", "실망", "불편", "비싸", "느리"
        ]

    def get_reputation(self, tool_name: str) -> float:
        """도구의 평판 점수 반환 (0~1)

        Web Search를 통해 리뷰를 검색하고 긍정/부정 멘션을 분석합니다.

        Args:
            tool_name: AI 도구명 (예: "ChatGPT")

        Returns:
            평판 점수 (0.0 ~ 1.0)
            - 1.0: 매우 긍정적
            - 0.5: 중립
            - 0.0: 매우 부정적
            - API 미설정 시: 0.7 (기본값)
        """
        if not self.enabled:
            # API 설정이 없으면 중립적 점수 반환
            return 0.7

        try:
            # 검색 쿼리 생성
            query = f"{tool_name} AI tool review"

            # Google Custom Search 실행
            result = self.google_service.cse().list(
                q=query,
                cx=self.search_engine_id,
                num=5  # 상위 5개 결과만
            ).execute()

            # 검색 결과 분석
            items = result.get("items", [])
            if not items:
                print(f"[ReputationEvaluator] '{tool_name}': 검색 결과 없음")
                return 0.6  # 기본값

            # 텍스트 추출 (제목 + 스니펫)
            texts = []
            for item in items:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                texts.append(title + " " + snippet)

            combined_text = " ".join(texts).lower()

            # 키워드 카운트
            positive_count = sum(
                len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', combined_text))
                for keyword in self.positive_keywords
            )

            negative_count = sum(
                len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', combined_text))
                for keyword in self.negative_keywords
            )

            # 점수 계산
            total = positive_count + negative_count
            if total == 0:
                # 키워드가 없으면 중립
                score = 0.6
            else:
                # 긍정 비율 (0.5 ~ 1.0 범위로 조정)
                score = 0.5 + (positive_count / total) * 0.5

            print(f"[ReputationEvaluator] '{tool_name}': 점수 {score:.2f} (긍정: {positive_count}, 부정: {negative_count})")
            return round(score, 2)

        except Exception as e:
            print(f"[ReputationEvaluator] '{tool_name}' 평판 조회 실패: {e}")
            return 0.6  # 에러 시 중립 점수


    def get_reputation_batch(self, tool_names: list) -> dict:
        """여러 도구의 평판을 일괄 조회

        Args:
            tool_names: 도구명 리스트

        Returns:
            {tool_name: reputation_score} 딕셔너리
        """
        results = {}
        for name in tool_names:
            results[name] = self.get_reputation(name)
        return results
