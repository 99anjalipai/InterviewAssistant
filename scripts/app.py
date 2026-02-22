import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from utils import extract_text
from rag_engine import (
    generate_interview_questions,
    generate_cover_letter,
    load_candidate_vectorstore,
    get_jd_vectorstore_from_text,
)

st.set_page_config(page_title="Job2Prep", layout="centered")

st.title("📄 Job2Prep — RAG-powered Interview + Cover Letter Generator")

# --- Load candidate DB once per session ---
if "candidate_vs" not in st.session_state:
    try:
        st.session_state["candidate_vs"] = load_candidate_vectorstore()
    except Exception as e:
        st.error(f"❌ Could not load candidate vector DB: {e}")
        st.stop()

uploaded_file = st.file_uploader("Upload a Job Description (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    try:
        jd_text = extract_text(uploaded_file)
        st.subheader("📌 Extracted Job Description:")
        with st.expander("Click to view extracted text"):
            st.text(jd_text)

        # --- Build / cache JD mini-index once per JD ---
        # If user uploads a different file, rebuild. Streamlit doesn't give a stable file hash
        # easily, so we use file name + size as a lightweight key.
        file_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("jd_file_key") != file_key:
            st.session_state["jd_file_key"] = file_key
            st.session_state["jd_text"] = jd_text
            st.session_state["jd_vs"] = get_jd_vectorstore_from_text(jd_text)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎯 Generate Interview Questions"):
                with st.spinner("Generating interview questions..."):
                    questions = generate_interview_questions(
                        jd_text=st.session_state["jd_text"],
                        jd_vs=st.session_state["jd_vs"],
                        candidate_vs=st.session_state["candidate_vs"],
                    )
                    st.subheader("🧠 Interview Questions")
                    st.write(questions)

        with col2:
            if st.button("✍️ Generate Cover Letter"):
                with st.spinner("Drafting your cover letter..."):
                    cover_letter = generate_cover_letter(
                        jd_text=st.session_state["jd_text"],
                        jd_vs=st.session_state["jd_vs"],
                        candidate_vs=st.session_state["candidate_vs"],
                    )
                    st.subheader("📬 Cover Letter")
                    st.write(cover_letter)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("Please upload a job description to get started.")