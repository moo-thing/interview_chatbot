import os # 운영 체제에서 사용할 수 있는 기능 제공 라이브러리
from fastapi import UploadFile, HTTPException
# ㄴ FastAPI: 웹 프레임워크, UploadFile: 파일 업로드 처리, File: 파일 업로드 필드, Form: 폼 데이터 처리
# FastAPI 속도가 빠름 + 비동기 지원 + 코드 간결 = 처리속도 증가
# Swagger 자동 생성! hhtp://localhost:8000/docs 에서 API 테스트 가능
# React ➡️ FastAPI ➡️ LLM ➡️ VectorDB 구조
from langchain_community.document_loaders import PyPDFLoader # PDF에서 텍스트 추출, 문서화

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