"""
Tools - Calculator, Time, Google Search 도구 정의
"""
import os
from typing import List, Dict, Optional, Any
from datetime import datetime

from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


# ==================== Calculator Tool ====================

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
    도구 리스트에서 비용 계산 (내부 함수)

    Args:
        tools: 도구 정보 딕셔너리 리스트

    Returns:
        비용 정보 딕셔너리
    """
    total_monthly = 0
    breakdown = []

    for tool in tools:
        price = tool.get('monthly_price', 0)
        name = tool.get('name', 'Unknown')
        total_monthly += price
        breakdown.append({
            "name": name,
            "monthly_price": price
        })

    return {
        "total_monthly": total_monthly,
        "total_yearly": total_monthly * 12,
        "breakdown": breakdown
    }


# ==================== Time Tool ====================

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


def get_current_time() -> str:
    """현재 시간 반환"""
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


# ==================== Google Search ====================

class GoogleSearchTool:
    """Google Custom Search API를 활용한 검색 도구"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self._service = None

    @property
    def is_available(self) -> bool:
        """API 사용 가능 여부"""
        return bool(self.api_key and self.search_engine_id)

    def _get_service(self):
        """Google API 서비스 객체 생성 (지연 로딩)"""
        if self._service is None and self.is_available:
            try:
                from googleapiclient.discovery import build
                self._service = build(
                    "customsearch", "v1",
                    developerKey=self.api_key
                )
            except Exception as e:
                print(f"Google Search API 초기화 실패: {e}")
                return None
        return self._service

    def search(
        self,
        query: str,
        num_results: int = 3,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        Google Custom Search 실행

        Args:
            query: 검색 쿼리
            num_results: 결과 수 (최대 10)
            category: AI 도구 카테고리 (검색어 최적화용)

        Returns:
            검색 결과 리스트
        """
        if not self.is_available:
            print("Google Search API가 설정되지 않았습니다.")
            return []

        service = self._get_service()
        if not service:
            return []

        # 쿼리 최적화
        optimized_query = self._optimize_query(query, category)

        try:
            result = service.cse().list(
                q=optimized_query,
                cx=self.search_engine_id,
                num=min(num_results, 10)
            ).execute()

            search_results = []
            for item in result.get('items', []):
                search_results.append({
                    "name": item.get('title', ''),
                    "description": item.get('snippet', ''),
                    "url": item.get('link', ''),
                    "source": "google_search",
                    "score": 0.5  # Google 검색 결과는 기본 점수 0.5
                })

            return search_results

        except Exception as e:
            print(f"Google Search 오류: {e}")
            return []

    def _optimize_query(self, query: str, category: Optional[str] = None) -> str:
        """검색 쿼리 최적화"""
        optimized = f"{query} AI tool"

        # 카테고리별 키워드 추가
        category_keywords = {
            "text-generation": "chatbot text generation",
            "image-generation": "AI image generator art",
            "video-generation": "AI video creation",
            "audio-generation": "AI voice TTS music",
            "code-generation": "AI coding assistant",
            "productivity": "AI productivity tool",
            "design": "AI design tool",
            "research": "AI research tool"
        }

        if category and category in category_keywords:
            optimized += f" {category_keywords[category]}"

        return optimized


# 싱글톤 인스턴스
google_search = GoogleSearchTool()


def web_search(query: str, category: Optional[str] = None) -> List[Dict]:
    """웹 검색 래퍼 함수"""
    return google_search.search(query, category=category)


# ==================== 통합 검색 함수 ====================

def hybrid_search(
    memory_manager,
    query: str,
    k: int = 5,
    threshold: float = 0.7,
    category: Optional[str] = None,
    use_web_fallback: bool = True,
    include_pdf: bool = True
) -> tuple[List[Dict], bool]:
    """
    하이브리드 검색: RAG (JSON + PDF) + Web Search

    Args:
        memory_manager: MemoryManager 인스턴스
        query: 검색 쿼리
        k: 반환할 최대 결과 수
        threshold: 유사도 임계값
        category: 카테고리 필터
        use_web_fallback: 웹 검색 폴백 사용 여부
        include_pdf: PDF 지식베이스 검색 포함 여부

    Returns:
        (검색 결과 리스트, fallback 발동 여부)
    """
    all_results = []

    # 1. JSON 도구 검색 (ChromaDB - ai_tools 컬렉션)
    rag_results, should_fallback = memory_manager.search_tools(
        query=query,
        k=k,
        threshold=threshold,
        category=category
    )

    # source 표시 추가
    for r in rag_results:
        r["source"] = "json"
    all_results.extend(rag_results)

    # 2. PDF 지식베이스 검색 (ChromaDB - pdf_knowledge 컬렉션)
    if include_pdf:
        pdf_results = memory_manager.search_pdf_knowledge(
            query=query,
            k=3,
            threshold=0.03
        )
        all_results.extend(pdf_results)

    # 3. Fallback 조건 확인 및 웹 검색
    if should_fallback and use_web_fallback and google_search.is_available:
        print(f"RAG 결과 부족 (threshold: {threshold}), 웹 검색 실행...")
        web_results = google_search.search(query, num_results=3, category=category)

        # source 표시 추가
        for wr in web_results:
            wr["source"] = "web"

        # 결과 병합 (중복 제거)
        existing_names = {r.get('name', '').lower() for r in all_results if r.get('name')}
        for wr in web_results:
            if wr.get('name', '').lower() not in existing_names:
                all_results.append(wr)

    # 점수순 정렬
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    return all_results[:k], should_fallback


# ==================== 도구 리스트 ====================

def get_all_tools() -> List:
    """LangChain 도구 리스트 반환"""
    return [
        calculate_subscription_cost,
        check_tool_freshness
    ]


# 테스트용 코드
if __name__ == "__main__":
    # Calculator 테스트
    result = calculate_subscription_cost.invoke({
        "tool_names": ["ChatGPT", "Midjourney", "ElevenLabs"],
        "tool_prices": [20.0, 30.0, 22.0]
    })
    print(result)

    print("\n" + "=" * 50 + "\n")

    # Time 테스트
    result = check_tool_freshness.invoke({
        "tool_name": "ChatGPT",
        "updated_date": "2024-11-01"
    })
    print(result)

    print("\n" + "=" * 50 + "\n")

    # 현재 시간
    print(f"현재 시간: {get_current_time()}")
