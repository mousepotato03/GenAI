"""
AI 101 - 지능형 AI 도구 추천 에이전트
엔트리포인트
"""
import uvicorn
from app.api.routes import create_app
from core.config import HOST, PORT

app = create_app()

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"AI 101 에이전트 서버 시작")
    print(f"URL: http://localhost:{PORT}")
    print(f"{'='*50}\n")

    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
