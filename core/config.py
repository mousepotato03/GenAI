"""
Core Config - 전역 설정 및 상수 정의
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM 설정
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# RAG 설정
SIMILARITY_THRESHOLD = 0.7
MAX_TOOL_CALLS_PER_TASK = 3

# 저장소 경로
DB_PATH = os.getenv("DB_PATH", "./db")
DATA_PATH = os.getenv("DATA_PATH", "./data")

# JSON 데이터 경로
TOOLS_JSON_PATH = os.path.join(DATA_PATH, "ai_tools_2025.json")

# 서버 설정
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 7860))
