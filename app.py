import streamlit as st
from utils import extract_text
from rag_engine import generate_interview_questions, generate_cover_letter

st.set_page_config(page_title="Job2Prep", layout="centered")

st.title("📄 Job2Prep — RAG-powered Interview + Cover Letter Generator")

uploaded_file = st.file_uploader("Upload a Job Description (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
        jd_text = extract_text(uploaded_file)
        st.subheader("📌 Extracted Job Description:")
        with st.expander("Click to view extracted text"):
            st.text(jd_text)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎯 Generate Interview Questions"):
                with st.spinner("Generating interview questions..."):
                    questions = generate_interview_questions(jd_text)
                    st.subheader("🧠 Interview Questions")
                    st.write(questions)

        with col2:
            if st.button("✍️ Generate Cover Letter"):
                with st.spinner("Drafting your cover letter..."):
                    cover_letter = generate_cover_letter(jd_text)
                    st.subheader("📬 Cover Letter")
                    st.write(cover_letter)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("Please upload a job description to get started.")
