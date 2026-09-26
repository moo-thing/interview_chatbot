from langchain_community.document_loaders import PyPDFLoader # PDF에서 텍스트 추출, 문서화
from langchain_text_splitters import RecursiveCharacterTextSplitter # 긴 텍스트 쪼갬
from langchain_community.vectorstores import Chroma # 벡터 데이터베이스, 유사도 검색 지원


def create_vectorstore(pdf_path, embeddings, db_path):
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
        persist_directory=db_path
    )

    return vectordb