# prompts.py

from langchain.prompts import PromptTemplate

INTERVIEW_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
You're a technical recruiter. Based on the job description below, generate:
- 5 technical interview questions
- 3 behavioral interview questions

Job Description:
{context}
"""
)

COVER_LETTER_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
You're a career coach. Based on the job description below, write:
1. A professional cover letter introduction
2. A paragraph explaining why the candidate is a strong fit

Job Description:
{context}
"""
)
