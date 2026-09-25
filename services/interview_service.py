from models.schemas import InterviewQuestions


def generate_interview_questions(
    ai_model,
    context,
    job_text,
    skill_analysis
):
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

    return result