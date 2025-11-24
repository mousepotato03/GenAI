"""
AI 101 시스템 통합 시작 스크립트

API 서버와 Gradio UI를 순차적으로 실행합니다.
"""

import os
import sys
import time
import subprocess
import requests
import signal
import socket
from pathlib import Path

# 프로세스 저장
api_process = None
ui_process = None

def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """사용 가능한 포트 찾기

    Args:
        start_port: 시작 포트 번호
        max_attempts: 최대 시도 횟수

    Returns:
        사용 가능한 포트 번호
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue

    raise RuntimeError(f"사용 가능한 포트를 찾을 수 없습니다 (시작: {start_port}, 시도: {max_attempts})")

def cleanup(signum=None, frame=None):
    """프로세스 정리"""
    print("\n\n" + "="*60)
    print("Shutting down...")
    print("="*60)

    if ui_process:
        print("UI 서버 종료 중...")
        ui_process.terminate()
        try:
            ui_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ui_process.kill()

    if api_process:
        print("API 서버 종료 중...")
        api_process.terminate()
        try:
            api_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_process.kill()

    print("All servers stopped.")
    sys.exit(0)

def check_api_health(url: str, max_retries: int = 30, delay: float = 2.0) -> bool:
    """API 서버 헬스 체크

    Args:
        url: API 서버 URL
        max_retries: 최대 재시도 횟수
        delay: 재시도 간격 (초)

    Returns:
        서버가 준비되면 True, 아니면 False
    """
    print(f"Waiting for API server to be ready...", end="", flush=True)

    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/", timeout=2)
            if response.status_code == 200:
                print(" ✅")
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass

        print(".", end="", flush=True)
        time.sleep(delay)

    print(" ❌")
    return False

def main():
    """메인 실행 함수"""
    global api_process, ui_process

    # Signal handler 등록
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 프로젝트 루트로 이동
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Python 실행 파일 경로
    python_cmd = sys.executable

    # 사용 가능한 포트 찾기
    api_host = os.getenv("API_HOST", "0.0.0.0")
    ui_host = os.getenv("UI_HOST", "0.0.0.0")

    # 환경 변수에서 포트를 지정하지 않았다면 자동으로 찾기
    if os.getenv("API_PORT"):
        api_port = int(os.getenv("API_PORT"))
    else:
        api_port = find_available_port(8000)
        print(f"✓ API 서버용 포트 자동 할당: {api_port}")

    if os.getenv("UI_PORT"):
        ui_port = int(os.getenv("UI_PORT"))
    else:
        ui_port = find_available_port(7860)
        print(f"✓ UI 서버용 포트 자동 할당: {ui_port}")

    api_url = f"http://localhost:{api_port}"

    print()
    print("="*60)
    print("Starting AI 101 System...")
    print("="*60)
    print()

    # 환경 변수 설정 (PYTHONPATH + 포트)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["API_HOST"] = api_host
    env["API_PORT"] = str(api_port)
    env["API_URL"] = api_url
    env["UI_HOST"] = ui_host
    env["UI_PORT"] = str(ui_port)

    # 1. API 서버 시작
    print(f"[1/2] Starting API Server on port {api_port}...")
    try:
        api_process = subprocess.Popen(
            [python_cmd, "src/api/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )

        # API 서버가 준비될 때까지 대기
        if not check_api_health(api_url):
            print("❌ API 서버 시작 실패")
            print("\n로그 확인:")
            if api_process.poll() is not None:
                stdout, stderr = api_process.communicate()
                if stderr:
                    print(stderr.decode('utf-8', errors='ignore'))
            cleanup()
            return

    except Exception as e:
        print(f"❌ API 서버 시작 실패: {e}")
        cleanup()
        return

    # 2. UI 서버 시작
    print(f"[2/2] Starting Gradio UI on port {ui_port}...")
    try:
        ui_process = subprocess.Popen(
            [python_cmd, "src/ui/app.py"],
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )

        # UI 서버 시작 대기 (짧게)
        time.sleep(3)

        if ui_process.poll() is not None:
            print("❌ UI 서버 시작 실패")
            stdout, stderr = ui_process.communicate()
            if stderr:
                print(stderr.decode('utf-8', errors='ignore'))
            cleanup()
            return

    except Exception as e:
        print(f"❌ UI 서버 시작 실패: {e}")
        cleanup()
        return

    # 성공 메시지
    print()
    print("="*60)
    print("🚀 System Ready!")
    print("="*60)
    print(f"📡 API Server:  {api_url}")
    print(f"🎨 Gradio UI:   http://localhost:{ui_port}")
    print("="*60)
    print()
    print("💡 브라우저에서 http://localhost:{} 로 접속하세요".format(ui_port))
    print("⏹️  종료하려면 Ctrl+C를 누르세요")
    print()

    # 프로세스 모니터링
    try:
        while True:
            # API 서버 체크
            if api_process.poll() is not None:
                print("\n❌ API 서버가 예기치 않게 종료되었습니다")
                cleanup()
                return

            # UI 서버 체크
            if ui_process.poll() is not None:
                print("\n❌ UI 서버가 예기치 않게 종료되었습니다")
                cleanup()
                return

            time.sleep(1)

    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
