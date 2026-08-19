# AI Interview

이력서와 채용공고를 기반으로 맞춤형 기술 면접 질문을 생성하는 LLM 프로젝트입니다.

## 주요 기능

- 이력서 PDF 업로드
- 채용공고 URL 입력
- 채용공고 크롤링
- PDF 텍스트 추출
- OpenAI Embeddings를 이용한 임베딩
- Chroma를 이용한 Vector DB 구축
- RAG 기반 이력서 정보 검색
- 이력서와 채용공고 기술 스택 비교
- 부족한 기술을 기반으로 맞춤형 면접 질문 생성

## 기술 스택

- Python
- FastAPI
- LangChain
- OpenAI API
- Chroma
- BeautifulSoup
- PyPDFLoader

## 실행 방법

### 1. 가상환경 생성

```bash
conda create -n ai_interview python=3.11