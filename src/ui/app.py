"""
Gradio UI

AI 101 에이전트 시스템의 사용자 인터페이스입니다.
채팅 인터페이스와 계획 승인 UI를 제공합니다.
"""

import os
import requests
import gradio as gr
from typing import List, Dict

# API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")


class ChatInterface:
    """채팅 인터페이스 관리 클래스"""

    def __init__(self):
        self.session_id = None
        self.current_plan = None
        self.waiting_approval = False

    def chat(self, message: str, history: List[Dict]):
        """사용자 메시지 처리

        Args:
            message: 사용자 입력
            history: 대화 히스토리 (Gradio 6.0 딕셔너리 형식)

        Returns:
            업데이트된 히스토리, 승인 UI 표시 여부, 계획 텍스트, 승인 버튼들 표시 여부
        """
        if not message.strip():
            return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        try:
            # 1. API 호출
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "query": message,
                    "session_id": self.session_id
                },
                timeout=300  # 5분 (초기 쿼리 및 Plan 생성)
            )

            if response.status_code != 200:
                error_msg = f"API 에러: {response.status_code}"
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": f"❌ {error_msg}"})
                return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            data = response.json()

            # 2. 세션 ID 저장
            self.session_id = data["session_id"]

            # 3. 상태별 처리
            if data["status"] == "awaiting_approval":
                # Plan 승인 대기
                self.current_plan = data["plan"]
                self.waiting_approval = True

                # 계획 포맷팅
                plan_text = self._format_plan(data["plan"])

                # 히스토리에 질문만 추가 (응답은 승인 후)
                history.append({"role": "user", "content": message})

                return (
                    history,
                    gr.update(visible=True),  # approval_group
                    plan_text,               # plan_display
                    gr.update(visible=True),  # approve_btn
                    gr.update(visible=True),  # reject_btn
                    gr.update(visible=False)  # feedback_box
                )

            elif data["status"] == "completed":
                # 최종 응답
                self.waiting_approval = False
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": data["response"]})

                return (
                    history,
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

            elif data["status"] == "error":
                # 에러
                error_msg = data.get("error", "알 수 없는 에러")
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": f"❌ 에러 발생:\n{error_msg}"})

                return (
                    history,
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

            else:
                # 기타 상태
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": f"처리 중... (상태: {data['status']})"})
                return (
                    history,
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

        except requests.Timeout:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "❌ 요청 시간 초과 (300초)"})
            return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ 에러: {str(e)}"})
            return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def approve(self, history: List[Dict]):
        """계획 승인

        Args:
            history: 대화 히스토리

        Returns:
            업데이트된 히스토리, UI 업데이트들
        """
        if not self.session_id or not self.waiting_approval:
            return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        try:
            # API 호출
            response = requests.post(
                f"{API_URL}/approve",
                json={
                    "session_id": self.session_id,
                    "approved": True
                },
                timeout=600  # 10분 (전체 워크플로우 실행 시간)
            )

            if response.status_code != 200:
                error_msg = f"API 에러: {response.status_code}"
                history.append({"role": "assistant", "content": f"❌ {error_msg}"})
                return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            data = response.json()

            # 최종 응답 표시
            if data["status"] == "completed":
                self.waiting_approval = False
                history.append({"role": "assistant", "content": data["response"]})

                return (
                    history,
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            else:
                history.append({"role": "assistant", "content": f"처리 중... (상태: {data['status']})"})
                return (
                    history,
                    gr.update(visible=False),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

        except requests.Timeout:
            history.append({"role": "assistant", "content": "❌ 처리 시간 초과 (600초)"})
            return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        except Exception as e:
            history.append({"role": "assistant", "content": f"❌ 에러: {str(e)}"})
            return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def reject(self, history: List[Dict]):
        """계획 거절 - 피드백 입력창 표시

        Args:
            history: 대화 히스토리

        Returns:
            UI 업데이트들
        """
        return (
            gr.update(visible=False),  # approve_btn
            gr.update(visible=False),  # reject_btn
            gr.update(visible=True)    # feedback_box
        )

    def submit_feedback(self, feedback: str, history: List[Dict]):
        """피드백 제출 및 계획 재생성

        Args:
            feedback: 사용자 피드백
            history: 대화 히스토리

        Returns:
            업데이트된 히스토리, UI 업데이트들
        """
        if not self.session_id or not self.waiting_approval:
            return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        if not feedback.strip():
            feedback = "다시 생각해주세요"

        try:
            # API 호출
            response = requests.post(
                f"{API_URL}/approve",
                json={
                    "session_id": self.session_id,
                    "approved": False,
                    "feedback": feedback
                },
                timeout=300  # 5분 (Plan 재생성)
            )

            if response.status_code != 200:
                error_msg = f"API 에러: {response.status_code}"
                return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            data = response.json()

            # 새 계획 표시
            if data["status"] == "awaiting_approval":
                self.current_plan = data["plan"]
                plan_text = self._format_plan(data["plan"])

                return (
                    history,
                    gr.update(visible=True),  # approval_group
                    plan_text,               # plan_display
                    gr.update(visible=True),  # approve_btn
                    gr.update(visible=True),  # reject_btn
                    gr.update(visible=False)  # feedback_box
                )

        except Exception as e:
            history.append({"role": "assistant", "content": f"❌ 에러: {str(e)}"})

        return history, gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def _format_plan(self, plan: List[dict]) -> str:
        """계획을 보기 좋게 포맷

        Args:
            plan: 계획 리스트

        Returns:
            Markdown 형식의 계획
        """
        text = "## 📋 실행 계획\n\n"
        for i, task in enumerate(plan, 1):
            text += f"{i}. **{task['description']}**\n"
            text += f"   - 카테고리: `{task['category']}`\n\n"

        text += "\n---\n\n"
        text += "👆 위 계획이 괜찮으신가요?\n\n"
        text += "- **승인**: 계획대로 진행합니다\n"
        text += "- **수정 요청**: 원하시는 변경사항을 알려주세요"

        return text


def create_ui():
    """Gradio 인터페이스 생성

    Returns:
        Gradio Blocks 인터페이스
    """
    interface = ChatInterface()

    with gr.Blocks(
        title="AI 101 - AI 도구 추천 에이전트"
    ) as demo:
        # 헤더
        gr.Markdown("""
        # 🤖 AI 101
        ### 당신의 아이디어를 실행 가능한 단계로 분해하고, 최적의 AI 도구를 추천합니다!
        """)

        # 채팅 인터페이스
        chatbot = gr.Chatbot(
            height=500,
            label="대화",
            show_label=True
        )

        msg = gr.Textbox(
            placeholder="무엇을 만들고 싶으신가요? (예: 미스테리 유튜브 쇼츠 만들기)",
            label="메시지",
            lines=2
        )

        with gr.Row():
            submit = gr.Button("전송", variant="primary", scale=2)
            clear = gr.Button("초기화", scale=1)

        # Plan 승인 UI (초기에는 숨김)
        with gr.Group(visible=False) as approval_group:
            plan_display = gr.Markdown("")

            with gr.Row():
                approve_btn = gr.Button("✅ 승인하고 진행", variant="primary", visible=True)
                reject_btn = gr.Button("✏️ 수정 요청", visible=True)

            feedback_box = gr.Textbox(
                placeholder="어떻게 수정해드릴까요? (예: 더 구체적으로 나눠주세요)",
                label="수정 요청 사항",
                lines=2,
                visible=False
            )

            submit_feedback_btn = gr.Button("피드백 제출", visible=False)

        # 예제
        gr.Examples(
            examples=[
                "유튜브 미스테리 쇼츠를 만들고 싶어. 시나리오부터 영상, 더빙까지 전부",
                "블로그 글을 쓰고 싶은데, 주제는 AI 활용법이야. 썸네일도 만들어줘",
                "회사 발표용 PPT를 만들고 싶어. 디자인도 예쁘게",
                "Python 웹 크롤러를 만들고 싶어. 코드 작성부터 디버깅까지"
            ],
            inputs=msg,
            label="예제 질문"
        )

        # 푸터
        gr.Markdown("""
        ---
        **사용 방법**:
        1. 하고 싶은 작업을 자유롭게 입력하세요
        2. AI가 단계별 계획을 생성합니다
        3. 계획을 확인하고 승인하거나 수정 요청하세요
        4. 승인 후 각 단계별 추천 도구와 사용 가이드를 받으세요!
        """)

        # 이벤트 핸들러

        # 메시지 전송
        submit.click(
            fn=interface.chat,
            inputs=[msg, chatbot],
            outputs=[chatbot, approval_group, plan_display, approve_btn, reject_btn, feedback_box]
        ).then(
            fn=lambda: "",
            outputs=msg
        )

        # Enter 키로 전송
        msg.submit(
            fn=interface.chat,
            inputs=[msg, chatbot],
            outputs=[chatbot, approval_group, plan_display, approve_btn, reject_btn, feedback_box]
        ).then(
            fn=lambda: "",
            outputs=msg
        )

        # 계획 승인
        approve_btn.click(
            fn=interface.approve,
            inputs=[chatbot],
            outputs=[chatbot, approval_group, plan_display, approve_btn, reject_btn, feedback_box]
        )

        # 계획 거절 (피드백 입력창 표시)
        reject_btn.click(
            fn=interface.reject,
            inputs=[chatbot],
            outputs=[approve_btn, reject_btn, feedback_box]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=submit_feedback_btn
        )

        # 피드백 제출
        submit_feedback_btn.click(
            fn=interface.submit_feedback,
            inputs=[feedback_box, chatbot],
            outputs=[chatbot, approval_group, plan_display, approve_btn, reject_btn, feedback_box]
        ).then(
            fn=lambda: (gr.update(visible=False), gr.update(value="")),
            outputs=[submit_feedback_btn, feedback_box]
        )

        # 초기화
        clear.click(
            fn=lambda: ([], None),
            outputs=[chatbot, msg]
        ).then(
            fn=lambda: (gr.update(visible=False), "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
            outputs=[approval_group, plan_display, approve_btn, reject_btn, feedback_box]
        )

    return demo


# ===== 서버 실행 =====
if __name__ == "__main__":
    # API 서버 확인
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        print(f"✅ API 서버 연결 확인: {API_URL}")
    except:
        print(f"⚠️  경고: API 서버에 연결할 수 없습니다: {API_URL}")
        print(f"API 서버를 먼저 실행해주세요: python src/api/main.py")

    # Gradio UI 실행
    demo = create_ui()

    host = os.getenv("UI_HOST", "0.0.0.0")
    port = int(os.getenv("UI_PORT", "7860"))

    print(f"\n{'='*60}")
    print(f"AI 101 Gradio UI 시작")
    print(f"주소: http://{host}:{port}")
    print(f"{'='*60}\n")

    demo.launch(
        server_name=host,
        server_port=port,
        share=False
    )
