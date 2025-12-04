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


# ==================== RAG 도구 (LLM 바인딩용) ====================

# 메모리 매니저 싱글톤 (지연 로딩)
_memory_manager = None


def _get_memory_manager():
    """메모리 매니저 싱글톤 반환"""
    global _memory_manager
    if _memory_manager is None:
        from src.memory import MemoryManager
        _memory_manager = MemoryManager(persist_dir="./db")
    return _memory_manager


@tool
def retrieve_docs(query: str, category: Optional[str] = None) -> str:
    """
    AI 도구 지식베이스에서 관련 문서를 검색합니다.
    JSON(기본 도구 정보)과 PDF(최신 트렌드)를 하이브리드로 검색합니다.

    Args:
        query: 검색 쿼리 (예: "유튜브 쇼츠 제작 AI", "이미지 생성 도구")
        category: 도구 카테고리 필터 (선택사항)
            - text-generation: 텍스트 생성
            - image-generation: 이미지 생성
            - video-generation: 비디오 생성
            - audio-generation: 음성/음악 생성
            - code-generation: 코드 생성
            - productivity: 생산성 도구
            - design: 디자인 도구
            - research: 리서치 도구

    Returns:
        검색 결과 (JSON 문자열) - 도구명, 설명, 가격, 유사도 점수 포함
    """
    import json
    memory = _get_memory_manager()

    results, should_fallback = hybrid_search(
        memory_manager=memory,
        query=query,
        k=5,
        threshold=0.7,
        category=category,
        use_web_fallback=False,  # 웹 검색은 별도 도구로
        include_pdf=True
    )

    # 결과 포맷팅
    formatted_results = []
    for r in results:
        if r.get('source') == 'pdf':
            formatted_results.append({
                "type": "pdf_reference",
                "content": r.get('content', '')[:500],  # 500자 제한
                "filename": r.get('filename', ''),
                "page": r.get('page', 0),
                "score": r.get('score', 0)
            })
        else:
            formatted_results.append({
                "type": "ai_tool",
                "name": r.get('name', ''),
                "category": r.get('category', ''),
                "description": r.get('description', ''),
                "pricing": r.get('pricing', ''),
                "monthly_price": r.get('monthly_price', 0),
                "url": r.get('url', ''),
                "score": r.get('score', 0)
            })

    return json.dumps({
        "results": formatted_results,
        "should_fallback": should_fallback,
        "total_count": len(formatted_results)
    }, ensure_ascii=False, indent=2)


@tool
def read_memory(user_id: str) -> str:
    """
    사용자의 장기 메모리(선호도, 히스토리)를 읽어옵니다.

    Args:
        user_id: 사용자 ID

    Returns:
        사용자 프로필 정보 (JSON 문자열)
        - preferred_categories: 선호 카테고리
        - price_preference: 가격 선호도
        - interests: 관심 분야
        - skill_level: 기술 수준
    """
    import json
    memory = _get_memory_manager()
    profile = memory.load_user_profile(user_id)

    if profile:
        return json.dumps(profile, ensure_ascii=False, indent=2)
    else:
        return json.dumps({"message": "사용자 프로필이 없습니다.", "user_id": user_id}, ensure_ascii=False)


@tool
def write_memory(user_id: str, preferences: str) -> str:
    """
    사용자의 선호도를 장기 메모리에 저장합니다.

    Args:
        user_id: 사용자 ID
        preferences: 저장할 선호도 (JSON 문자열)
            예: {"preferred_categories": ["video-generation"], "price_preference": "무료선호"}

    Returns:
        저장 결과 메시지
    """
    import json
    memory = _get_memory_manager()

    try:
        prefs = json.loads(preferences)
        success = memory.save_user_profile(user_id, prefs)

        if success:
            return json.dumps({
                "status": "success",
                "message": f"사용자 {user_id}의 프로필이 저장되었습니다.",
                "saved_data": prefs
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "status": "error",
                "message": "프로필 저장에 실패했습니다."
            }, ensure_ascii=False)

    except json.JSONDecodeError as e:
        return json.dumps({
            "status": "error",
            "message": f"JSON 파싱 오류: {str(e)}"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"저장 오류: {str(e)}"
        }, ensure_ascii=False)


@tool
def google_search_tool(query: str, num_results: int = 3) -> str:
    """
    Google에서 최신 정보를 검색합니다.
    지식베이스에 없는 최신 AI 도구나 트렌드를 찾을 때 사용합니다.

    Args:
        query: 검색 쿼리 (예: "2024 최신 AI 영상 편집 도구")
        num_results: 반환할 결과 수 (기본 3개, 최대 10개)

    Returns:
        검색 결과 (JSON 문자열) - 제목, 설명, URL 포함
    """
    import json

    if not google_search.is_available:
        return json.dumps({
            "status": "error",
            "message": "Google Search API가 설정되지 않았습니다.",
            "results": []
        }, ensure_ascii=False)

    results = google_search.search(query, num_results=min(num_results, 10))

    return json.dumps({
        "status": "success",
        "query": query,
        "results": results,
        "total_count": len(results)
    }, ensure_ascii=False, indent=2)


# ==================== 도구 실행 헬퍼 ====================

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
        "check_tool_freshness": check_tool_freshness
    }

    tool_func = tools_map.get(tool_name)
    if tool_func:
        return tool_func.invoke(args)
    raise ValueError(f"알 수 없는 도구: {tool_name}")


# ==================== 도구 리스트 ====================

def get_all_tools() -> List:
    """LangChain 도구 리스트 반환 (LLM 바인딩용)"""
    return [
        retrieve_docs,
        read_memory,
        write_memory,
        google_search_tool,
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
