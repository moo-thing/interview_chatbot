import os # 운영 체제에서 사용할 수 있는 기능 제공 라이브러리
from fastapi import FastAPI, UploadFile, File, Form
# ㄴ FastAPI: 웹 프레임워크, UploadFile: 파일 업로드 처리, File: 파일 업로드 필드, Form: 폼 데이터 처리
# FastAPI 속도가 빠름 + 비동기 지원 + 코드 간결 = 처리속도 증가
# Swagger 자동 생성! hhtp://localhost:8000/docs 에서 API 테스트 가능
# React ➡️ FastAPI ➡️ LLM ➡️ VectorDB 구조
from langchain_openai import ChatOpenAI # OpenAI 연동
from langchain_openai import OpenAIEmbeddings # 텍스트 숫자 변환
from dotenv import load_dotenv # .env 파일에서 환경 변수 로드
from services.interview_service import generate_interview_questions
from services.job_service import crawl_job_posting
from services.skill_service import analyze_skills
from  services.vector_service import create_vectorstore

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

    # LLM아 이 형태로 대답해!
    
DB_PATH = "./chroma_db" # Chroma 벡터 DB 저장 경로

# post 방식으로 제출한 데이터 처리 API
@app.post("/generate_questions") # 서버주소/generate_questions 로 요청 시 처리해줌
# async 비동기 처리 가능 뒤에는 함수 정의
async def generate_questions( 
    resume: UploadFile = File(...), # 파일 형태 이력서 받음 / (...)은 필수 입력
    job_url : str = Form(...)
    ): # 채용 공고를 문자열 + url 가능하게 변경

    job_text = crawl_job_posting(job_url)

    # 파일 없을 시 생성, 있으면 무시
    os.makedirs("uploads", exist_ok=True)

    # 파일 저장 위치와 이름 설정
    upload_path = f"uploads/{resume.filename}"

    # 지정 경로에 업로드 파일 저장
    with open(upload_path, "wb") as f:
        f.write(await resume.read()) # 비동기 방식으로 파일 읽고 저장

    # 만들어둔 함수로 변수 설정
    vectordb = create_vectorstore(upload_path, embeddings, DB_PATH)

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

    skill_analysis = analyze_skills(
        ai_model,
        context,
        job_text
    )

    required_count = len(
        skill_analysis.required_skills
    )

    matched_count = len(
        skill_analysis.matched_skills
    )

    if required_count > 0:
        match_rate = round(
            matched_count / required_count * 100,
            1
        )
    else:
        match_rate = 0

    result = generate_interview_questions(
        ai_model,
        context,
        job_text,
        skill_analysis
    )

    return result