import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st

from rag_core import rag_pipeline


st.set_page_config(
    page_title="AIVLE School 학습 도우미",
    layout="wide"
)

st.title("AIVLE School 학습 도우미")
st.caption("업로드한 PDF를 근거로 질문에 답합니다.")

# 세션 상태 초기화
defaults = {
    "rag_chain": None,
    "messages": [],
    "active_hash": None,
    "active_name": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# API 키 설정 확인: 실제 키는 화면에 표시하지 않음
if not os.environ.get("OPENAI_API_KEY"):
    st.error("실행 환경에 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

with st.sidebar:
    st.header("문서 설정")

    uploaded_file = st.file_uploader(
        "백서 PDF 업로드",
        type=["pdf"]
    )

    uploaded_bytes = (
        uploaded_file.getvalue()
        if uploaded_file is not None
        else None
    )

    uploaded_hash = (
        hashlib.sha256(uploaded_bytes).hexdigest()
        if uploaded_bytes is not None
        else None
    )

    if st.button(
        "문서 적용",
        disabled=uploaded_file is None
    ):
        if not uploaded_bytes:
            st.warning("내용이 있는 PDF 파일을 선택해 주세요.")

        elif uploaded_hash == st.session_state.active_hash:
            st.info("이미 적용된 문서입니다.")

        else:
            # 기존 체인과 기록을 먼저 비움
            st.session_state.rag_chain = None
            st.session_state.active_hash = None
            st.session_state.active_name = None
            st.session_state.messages = []

            temp_path = None

            try:
                with st.spinner("문서를 읽고 검색 데이터를 만드는 중입니다."):
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=False
                    ) as temp_file:
                        temp_file.write(uploaded_bytes)
                        temp_path = Path(temp_file.name)

                    chain = rag_pipeline(temp_path)

                    st.session_state.rag_chain = chain
                    st.session_state.active_hash = uploaded_hash
                    st.session_state.active_name = uploaded_file.name

                st.success("문서 적용이 완료됐습니다.")

            except Exception:
                st.error(
                    "문서 적용에 실패했습니다. "
                    "PDF 본문, API 설정 및 실행 로그를 확인해 주세요."
                )

            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

    if st.session_state.active_name:
        st.caption(
            f"적용된 문서: {st.session_state.active_name}"
        )

    if st.button("대화 삭제"):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "문서 적용 시 임베딩 API, "
        "질문 시 답변 생성 API를 사용합니다."
    )

# 새 파일을 선택했지만 적용하지 않은 상태인지 확인
pending_document = (
    uploaded_file is not None
    and uploaded_hash != st.session_state.active_hash
)

if pending_document:
    st.info("선택한 PDF를 사용하려면 '문서 적용'을 눌러주세요.")

if st.session_state.rag_chain is None:
    st.info("왼쪽에서 백서 PDF를 업로드하고 '문서 적용'을 눌러주세요.")

# 기존 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력
question = st.chat_input(
    "백서에 대해 질문해 주세요.",
    disabled=(
        st.session_state.rag_chain is None
        or pending_document
    )
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("백서를 확인하는 중입니다."):
                answer = st.session_state.rag_chain.invoke({
                    "input": question
                })

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception:
            st.error(
                "답변 생성에 실패했습니다. "
                "API 설정 및 실행 로그를 확인해 주세요."
            )
