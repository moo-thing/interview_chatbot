import requests # HTTP 요청 처리 라이브러리
from bs4 import BeautifulSoup # HTML 파싱 라이브러리
from fastapi import HTTPException

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