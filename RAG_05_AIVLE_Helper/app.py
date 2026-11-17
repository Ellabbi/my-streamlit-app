import hashlib
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from rag_core import rag_pipeline

st.set_page_config(page_title="AIVLE School 학습 도우미", layout="wide")
st.title("AIVLE School 학습 도우미")
st.caption("PDF 기반 질의응답 · FAQ 및 추천 질문 · 세션별 문서 관리")

FAQ = {
    "지원·교육과정": [
        "AI 개발자 트랙과 DX 컨설턴트 트랙은 어떤 차이가 있나요?",
        "백서에 안내된 지원 자격과 지원 전 확인할 사항을 알려주세요.",
        "교육 전에 어떤 기초를 준비하면 좋을까요?"],
    "교육생활": [
        "출결 기준과 수료 기준을 구분하여 설명해주세요.",
        "수업 중 모르는 내용은 어디에 질문할 수 있나요?",
        "교안과 강의 다시보기는 어디에서 확인하나요?"],
    "프로젝트·진로": [
        "미니프로젝트와 빅프로젝트의 차이는 무엇인가요?",
        "프로젝트를 마친 뒤 무엇을 기록하면 좋을까요?",
        "포트폴리오 활용과 외부 공개 시 주의할 사항은 무엇인가요?"],
    "학습 전략": [
        "선배 후기에서 참고할 수 있는 학습 전략을 알려주세요.",
        "학습 내용을 이해하기 어려울 때 어떻게 복습하면 좋을까요?",
        "개인 학습 계획을 세울 때 점검할 항목은 무엇인가요?"]
}
FOLLOWUPS = ["방금 질문과 관련해 실천할 일을 체크리스트로 정리해주세요.",
             "방금 질문과 관련한 기준과 주의사항을 구분해주세요.",
             "방금 질문과 관련된 백서 내용을 초보자에게 쉽게 설명해주세요."]
DEFAULTS = {"documents_v2": {}, "active_doc_v2": None, "uploader_v2": 0}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value
registry = st.session_state.documents_v2

if not os.environ.get("OPENAI_API_KEY"):
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("Streamlit Secrets에 OPENAI_API_KEY를 설정해주세요.")
        st.stop()

with st.sidebar:
    st.header("문서 관리")
    st.caption("현재 세션에서만 보관됩니다. 새 세션이나 서버 재시작 후 다시 업로드해야 할 수 있습니다.")
    uploads = st.file_uploader("PDF 추가", type=["pdf"], accept_multiple_files=True,
                              key=f"pdf_upload_{st.session_state.uploader_v2}")
    if st.button("선택한 PDF 등록", disabled=not uploads):
        for uploaded in uploads:
            data = uploaded.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            if not data or len(data) > 20 * 1024 * 1024:
                st.warning(f"{uploaded.name}: 비어 있거나 20MB를 초과합니다.")
            elif not data[:1024].lstrip().startswith(b"%PDF-"):
                st.warning(f"{uploaded.name}: PDF 형식을 확인해주세요.")
            elif digest in registry:
                st.info(f"{uploaded.name}: 같은 내용의 문서가 이미 등록되어 있습니다.")
            elif len(registry) >= 5:
                st.warning("문서는 세션당 최대 5개까지 등록할 수 있습니다.")
            else:
                registry[digest] = {"name": uploaded.name, "data": data,
                    "registered": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "resource": None, "messages": [], "last_question": None}
                st.success(f"{uploaded.name}: 등록 완료")
        st.session_state.uploader_v2 += 1
        st.rerun()
    st.caption("등록만으로 API를 호출하지 않습니다. 검색 준비 시 임베딩 비용이 발생합니다.")

    if registry:
        chosen = st.selectbox("관리할 문서", list(registry),
            format_func=lambda key: registry[key]["name"], key="doc_choice_v2")
        doc = registry[chosen]
        st.caption(f"크기: {len(doc['data']) / 1024:,.1f} KB")
        st.caption(f"등록: {doc['registered']}")
        st.caption("상태: " + ("검색 준비 완료" if doc["resource"] else "등록됨 · 검색 준비 전"))
        if doc["resource"]:
            st.caption(f"Document {doc['resource']['document_count']}개 / 조각 {doc['resource']['chunk_count']}개")
        if st.button("이 문서로 대화 시작", type="primary"):
            if doc["resource"] is None:
                temp_path = None
                try:
                    with st.spinner("PDF 분석 및 임베딩 중입니다..."):
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
                            temp.write(doc["data"])
                            temp_path = Path(temp.name)
                        doc["resource"] = rag_pipeline(temp_path, return_resources=True)
                    st.session_state.active_doc_v2 = chosen
                    st.rerun()
                except Exception:
                    logging.exception("Document indexing failed")
                    st.error("문서 처리에 실패했습니다. PDF 본문과 API 설정을 확인해주세요.")
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
            else:
                st.session_state.active_doc_v2 = chosen
                st.rerun()
        st.download_button("원본 PDF 다운로드", doc["data"],
                           file_name=Path(doc["name"]).name, mime="application/pdf")
        with st.expander("추출 본문 미리보기"):
            if doc["resource"]:
                st.text(doc["resource"]["preview"])
            else:
                st.caption("검색 준비 후 본문을 확인할 수 있습니다.")
        confirmed = st.checkbox("선택한 문서와 해당 대화·검색 데이터를 삭제합니다.", key=f"delete_{chosen}")
        if st.button("선택 문서 삭제", disabled=not confirmed):
            try:
                if doc["resource"]:
                    doc["resource"]["store"].delete_collection()
            except Exception:
                logging.exception("Collection deletion failed")
                st.error("검색 데이터 삭제에 실패했습니다. 잠시 후 다시 시도해주세요.")
            else:
                del registry[chosen]
                if st.session_state.active_doc_v2 == chosen:
                    st.session_state.active_doc_v2 = None
                st.rerun()
    else:
        st.info("등록된 문서가 없습니다.")

active = registry.get(st.session_state.active_doc_v2)
ready = active is not None and active["resource"] is not None
st.subheader("FAQ 및 추천 질문")
st.caption("자주 궁금해할 질문의 예시입니다. 정답을 미리 고정하지 않고 현재 적용 문서에서 검색해 답합니다.")
category = st.selectbox("질문 주제", list(FAQ))
question = None
for index, label in enumerate(FAQ[category]):
    if st.button(label, key=f"faq_{category}_{index}", disabled=not ready):
        question = label

if ready:
    st.success(f"현재 대화 문서: {active['name']}")
    if active["last_question"]:
        with st.expander("이어서 물어보기"):
            for index, label in enumerate(FOLLOWUPS):
                if st.button(label, key=f"follow_{index}"):
                    question = (f"관련된 이전 질문: {active['last_question']}\n"
                                f"추가 요청: {label}")
    if st.button("현재 문서 대화 삭제"):
        active["messages"] = []
        active["last_question"] = None
        st.rerun()
    for message in active["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info("왼쪽에서 PDF를 등록한 뒤 '이 문서로 대화 시작'을 눌러주세요.")

typed = st.chat_input("현재 적용 문서에 대해 질문해주세요.", disabled=not ready)
if typed:
    question = typed
if question and ready:
    active["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("문서에서 답변을 찾는 중입니다..."):
                answer = active["resource"]["chain"].invoke({"input": question})
            st.markdown(answer)
            active["messages"].append({"role": "assistant", "content": answer})
            active["last_question"] = question if not question.startswith("관련된 이전 질문:") else active["last_question"]
        except Exception:
            logging.exception("Answer generation failed")
            st.error("답변 생성에 실패했습니다. API 설정 및 서버 로그를 확인해주세요.")
