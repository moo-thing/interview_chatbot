import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

app = FastAPI() # FastAPI 서버 생성

load_dotenv() # .env 파일에서 환경 변수 로드

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 모델 상호작용하는 LLM 생성
ai_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
# temperature = 0 심플하게 답변해줌, RAG, 검색, 분석에 적합
# 1 = 조금 창의적, 2 = 매우 창의적, 소설, 시, 아이디어 생성에 사용

embeddings = OpenAIEmbeddings()

DB_PATH = "./chroma_db" # Chroma 벡터 DB 저장 경로


def create_vectorstore(pdf_path):
    # PDF 경로 읽기, 텍스트 데이터 변환
    load_pdf = PyPDFLoader(pdf_path)
    docs = load_pdf.load()

    # LLM 문맥 제한 + 검색 효율 증가
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # 최대 길이 제한
        chunk_overlap=200 # 겹치는 부분 설정 문맥 이해 때문
    )

    # 문서 쪼개기
    chunks = splitter.split_documents(docs)

    # 쪼갠 텍스트 -> 임베딩 모델 -> 벡터화 -> Chroma DB 저장
    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_PATH
    )

    return vectordb


async def extract_text_from_file(upload_file: UploadFile) -> str:
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{upload_file.filename}"

    with open(file_path, "wb") as f:
        f.write(await upload_file.read())

    if file_path.lower().endswith(".pdf"):
        docs = PyPDFLoader(file_path).load()
        return "\n".join(doc.page_content for doc in docs)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# post 방식으로 제출한 데이터 처리 API
@app.post("/generate_questions") # 서버주소/generate_questions 로 요청 시 처리해줌
# async 비동기 처리 가능 뒤에는 함수 정의
async def generate_questions(
    resume: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    job_file: Optional[UploadFile] = File(None)
):
    if not job_description and not job_file:
        raise HTTPException(400, "job_description 또는 job_file 중 하나는 필요합니다")

    if job_file:
        job_text = await extract_text_from_file(job_file)
    else:
        job_text = job_description

    os.makedirs("uploads", exist_ok=True)
    upload_path = f"uploads/{resume.filename}"

    with open(upload_path, "wb") as f:
        f.write(await resume.read())

    vectordb = create_vectorstore(upload_path)

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 5}
    )

    context_docs = retriever.invoke(job_text)

    context = "\n".join(
        [doc.page_content for doc in context_docs]
    )

    prompt = f"""
당신은 시니어 기술 면접관이다.

지원자 이력서 정보:
{context}

채용공고:
{job_text}

기술 면접 질문 10개를 생성하라.

출력 형식:

1.
2.
3.
...
"""

    # AI 모델 답변이 result 변수에 담김
    result = ai_model.invoke(prompt)

    # FastAPI에서 딕셔너리 -> JSON 변환(자동!)
    return {
        "questions": result.content
    }