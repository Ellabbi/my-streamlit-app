from operator import itemgetter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_pipeline(filepath):
    # 1. PDF 읽기
    loader = PyMuPDF4LLMLoader(str(filepath))
    docs = loader.load()
    docs = [doc for doc in docs if doc.page_content.strip()]

    if not docs:
        raise ValueError("PDF에서 읽을 수 있는 본문이 없습니다.")

    print(f"1. PDF 읽기 완료: {len(docs)}개 Document")

    # 2. 문서 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    print(f"2. 문서 분할 완료: {len(chunks)}개 조각")

    # 3. 임베딩 모델
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 4. 벡터 저장소
    vectorstore = Chroma(
        collection_name="aivle_docs",
        embedding_function=embeddings
    )
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    print("3. 임베딩 및 벡터 저장 완료")

    # 5. 검색기
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # 6. 답변 규칙
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """당신은 AIVLE School DX 트랙 학습 도우미입니다.

아래 [참고 문서]만 근거로 질문에 답하세요.

답변 규칙:
1. 질문에 직접 답하고, 필요한 경우 항목별로 설명하세요.
2. 공식 안내, 개인 후기, 작성자의 학습 제안을 구분하세요.
3. 문서에 근거가 없으면
   "백서에서 해당 내용을 확인할 수 없습니다."라고 말하세요.
4. 지원 조건, 일정, 수료 기준처럼 변경될 수 있는 정보는
   해당 기수의 공식 공지 확인이 필요함을 안내하세요.
5. 문서에 없는 사실이나 수치를 만들지 마세요.
6. 친절하고 이해하기 쉬운 한국어로 답하세요.

[참고 문서]
{context}"""
        ),
        ("human", "{input}")
    ])

    # 7. 답변 모델
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # 8. RAG 체인
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("4. RAG 체인 구성 완료")

    return rag_chain
