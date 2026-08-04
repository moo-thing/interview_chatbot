import os # 운영 체제에서 사용할 수 있는 기능 제공 라이브러리
from typing import Optional # None 값 허용
from fastapi import FastAPI, UploadFile, File, Form
# ㄴ FastAPI: 웹 프레임워크, UploadFile: 파일 업로드 처리, File: 파일 업로드 필드, Form: 폼 데이터 처리
# FastAPI 속도가 빠름 + 비동기 지원 + 코드 간결 = 처리속도 증가
# Swagger 자동 생성! hhtp://localhost:8000/docs 에서 API 테스트 가능
# React ➡️ FastAPI ➡️ LLM ➡️ VectorDB 구조
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from langchain_openai import ChatOpenAI # OpenAI 연동
from langchain_openai import OpenAIEmbeddings # 텍스트 숫자 변환
from langchain_community.vectorstores import Chroma # 벡터 데이터베이스, 유사도 검색 지원
from langchain_community.document_loaders import PyPDFLoader # PDF에서 텍스트 추출, 문서화
from langchain_text_splitters import RecursiveCharacterTextSplitter # 긴 텍스트 쪼갬
from dotenv import load_dotenv # .env 파일에서 환경 변수 로드

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
    # 업로드 파일 텍스트 추출 함수(비동기)
    os.makedirs("uploads", exist_ok=True)
    # uploads 폴더 없으면 생성, 있을 시 무시
    file_path = f"uploads/{upload_file.filename}"
    # 업로드 파일 저장 경로 설정

    with open(file_path, "wb") as f:
        # PDF 는 바이너리 형식, wb 쓰는이유임.
        f.write(await upload_file.read())
        # 업로드 파일 읽음

    if file_path.lower().endswith(".pdf"):
        # 확장자 pdf인지 확인
        docs = PyPDFLoader(file_path).load()
        # PDF 파일 읽고 docs 객체들 만듦
        return "\n".join(doc.page_content for doc in docs)
        # PDF 각 페이지 내용 붙임

    # html, htm, txt 파일까지 OK
    elif file_path.lower().endswith((".html", ".htm", ".txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            # 텍스트나 HTML 파일 읽음
            return f.read()

    else:
        raise HTTPException(
            status_code = 400,
            detail = "PDF, HTML, TXT 파일만 지원하니 다시 한 번 확인해주세요."
        )
        

# post 방식으로 제출한 데이터 처리 API
@app.post("/generate_questions") # 서버주소/generate_questions 로 요청 시 처리해줌
# async 비동기 처리 가능 뒤에는 함수 정의
async def generate_questions( 
    resume: UploadFile = File(...), # 파일 형태 이력서 받음 / (...)은 필수 입력
    job_description: Optional[str] = Form(None),
    job_file: Optional[UploadFile] = File(None)
): # 채용 공고를 일반 텍스트로 받다가 파일 + 텍스트 가능하게 바꿈

    if not job_description and not job_file:
        raise HTTPException(400, "job_description 또는 job_file 중 하나는 필요합니다")
    # description, file 둘 중 하나 무조건 필요, 없을 시 400 에러


    if job_file:
        job_text = await extract_text_from_file(job_file)
        # file 업로드 시 텍스트 추출 후 변수 저장
    else:
        job_text = job_description
        # 파일 없을 시 str 타이핑 친 내용 저장
    # 변수 새로 만든 이유 : 항상 str만 사용하면 됨.(PDF, HTML, 직접 입력 신경쓸 필요 없음)

    # 파일 없을 시 생성, 있으면 무시
    os.makedirs("uploads", exist_ok=True)

    # 파일 저장 위치와 이름 설정
    upload_path = f"uploads/{resume.filename}"

    # 지정 경로에 업로드 파일 저장
    with open(upload_path, "wb") as f:
        f.write(await resume.read()) # 비동기 방식으로 파일 읽고 저장

    # 만들어둔 함수로 변수 설정
    vectordb = create_vectorstore(upload_path)

    # Chroma에 검색기능 추가 후 리트리버
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 5} # 5개 유사한 파일 가져옴
    )

    # 검색 수행 / 벡터 DB에서 유사한 문서 가져옴 / invoke는 str 형태로 받음
    context_docs = retriever.invoke(
        job_text
    ) # 채용공고 임베딩 후 유사한 문서 검색
    # 이력서에 기술을 서술해 놓고 채용공고에서 내가 가진 기술에 대한 공고를 

    # 하나의 문자열로 변환(한 줄씩 띄워진 상태)
    context = "\n".join(
        [doc.page_content for doc in context_docs]
    )

    # 이력서가 30페이지면 API에 다 보내기엔 데이터도 많이 잡아먹음
    # 질문 관련 부분만 추출 후 전달
    # 토큰 사용량 줄임, 속도 빠름, 답변 정확도 올라감

    prompt = f"""
당신은 시니어 기술 면접관이다.

지원자 이력서 정보:
{context}

채용공고:
{job_description}

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