from pydantic import BaseModel # None 값 허용

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