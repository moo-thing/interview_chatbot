from models.schemas import SkillAnalysis

def analyze_skills(ai_model, resume_text: str, job_text: str):
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