from langchain.prompts import PromptTemplate

INTERVIEW_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=(
        "You are an interview assistant.\n\n"
        "{context}\n\n"
        "Task: Generate 8 to 12 interview questions.\n"
        "Mix: technical, system design, behavioral.\n"
        "Also include 3 follow-up questions for the hardest technical question.\n"
    ),
)

COVER_LETTER_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=(
        "You are an interview assistant.\n\n"
        "{context}\n\n"
        "Task: Write a cover letter.\n"
        "Constraints:\n"
        "1) Exactly 3 short paragraphs.\n"
        "2) No em dashes and no hyphens.\n"
        "3) Do not claim tools, metrics, or experiences not in the candidate evidence.\n"
        "Tone: natural and conversational.\n"
    ),
)