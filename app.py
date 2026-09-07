import os # 운영 체제에서 사용할 수 있는 기능 제공 라이브러리
from pydantic import BaseModel # None 값 허용
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# ㄴ FastAPI: 웹 프레임워크, UploadFile: 파일 업로드 처리, File: 파일 업로드 필드, Form: 폼 데이터 처리
# FastAPI 속도가 빠름 + 비동기 지원 + 코드 간결 = 처리속도 증가
# Swagger 자동 생성! hhtp://localhost:8000/docs 에서 API 테스트 가능
# React ➡️ FastAPI ➡️ LLM ➡️ VectorDB 구조
from langchain_openai import ChatOpenAI # OpenAI 연동
from langchain_openai import OpenAIEmbeddings # 텍스트 숫자 변환
from langchain_community.vectorstores import Chroma # 벡터 데이터베이스, 유사도 검색 지원
from langchain_community.document_loaders import PyPDFLoader # PDF에서 텍스트 추출, 문서화
from langchain_text_splitters import RecursiveCharacterTextSplitter # 긴 텍스트 쪼갬
from dotenv import load_dotenv # .env 파일에서 환경 변수 로드
import requests # HTTP 요청 처리 라이브러리
from bs4 import BeautifulSoup # HTML 파싱 라이브러리

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
class SkillAnalysis(BaseModel):
    required_skills: list[str]
    resume_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]

class InterviewQuestion(BaseModel):
    questions: str
    intent : str
    evaluation_point : str
    answer_direction : str
    related_skill : str

class InterviewQuestions(BaseModel):
    questions : list[InterviewQuestion]
    
DB_PATH = "./chroma_db" # Chroma 벡터 DB 저장 경로

def crawl_job_posting(url: str) -> str:
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=10
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail="채용공고를 가져오지 못했습니다."
        )

    soup = BeautifulSoup(response.text, "html.parser")

    # 필요 없는 태그 제거
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    job_text = soup.get_text(
        separator="\n",
        strip=True
    )

    if not job_text:
        raise HTTPException(
            status_code=400,
            detail="채용공고 내용을 가져오지 못했습니다."
        )

    return job_text


def analyze_skills(resume_text: str, job_text: str):
    prompt = f"""
당신은 채용 분석 전문가입니다.

지원자 이력서:
{resume_text}

채용공고:
{job_text}

다음 내용을 분석하세요.

1. 채용공고에서 요구하는 기술을 추출하세요.
2. 이력서에서 지원자가 가지고 있는 기술을 추출하세요.
3. 두 기술 중 서로 일치하는 기술을 찾으세요.
4. 채용공고에는 있지만 이력서에는 없는 기술을 찾으세요.

기술 이름은 최대한 간결하게 작성하세요.
예:
Python, FastAPI, Docker, AWS, Kubernetes
"""

    structured_model = ai_model.with_structured_output(
        SkillAnalysis
    )

    result = structured_model.invoke(prompt)

    return result

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

    skill_analysis = analyze_skills(
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

    prompt = f"""
당신은 시니어 기술 면접관이다.

지원자 이력서 관련 정보:
{context}

채용공고:
{job_text}

기술 분석 결과:

보유 기술:
{skill_analysis.resume_skills}

일치하는 기술:
{skill_analysis.matched_skills}

부족한 기술:
{skill_analysis.missing_skills}

위 정보를 기반으로 기술 면접 질문 10개를 생성하라.

특히 부족한 기술과 채용공고에서 중요하게 요구하는 기술을 중심으로 질문하라.

각 질문마다 다음 내용을 작성하라.

- question: 실제 면접 질문
- intent: 이 질문을 하는 이유
- evaluation_point: 면접관이 평가하려는 핵심 포인트
- answer_direction: 지원자가 답변할 때 포함하면 좋은 내용
- related_skill: 관련 기술

질문은 서로 중복되지 않게 작성하라.
"""

    structured_model = ai_model.with_structured_output(
        InterviewQuestions
    )
    result = structured_model.invoke(prompt)

    # 8. 결과 반환
    return {
        "match_rate": match_rate,
        "required_skills": skill_analysis.required_skills,
        "resume_skills": skill_analysis.resume_skills,
        "matched_skills": skill_analysis.matched_skills,
        "missing_skills": skill_analysis.missing_skills,
        "questions": result.questions
    }