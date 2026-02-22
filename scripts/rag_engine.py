import os
from typing import List, Tuple, Optional
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

from llm_groq import GroqLLM
from prompts import INTERVIEW_PROMPT, COVER_LETTER_PROMPT


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATE_DB_DIR = os.path.join("stores", "candidate_faiss")

# Small retrieval budgets for CPU
JD_K = 3
CAND_PER_JD_K = 1
MAX_CAND_DOCS = 4

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def get_llm() -> GroqLLM:
    if "groq_llm" not in st.session_state:
        st.session_state["groq_llm"] = GroqLLM(
            model="llama-3.1-8b-instant",
            max_tokens=700,
            temperature=0.3,
        )
    return st.session_state["groq_llm"]


def chunk_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.create_documents([text])


def get_embedder():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_jd_vectorstore_from_text(jd_text: str) -> FAISS:
    jd_chunks = chunk_text(jd_text)
    return FAISS.from_documents(jd_chunks, get_embedder())


def load_candidate_vectorstore() -> FAISS:
    if not os.path.exists(CANDIDATE_DB_DIR):
        raise FileNotFoundError(
            f"Candidate FAISS index not found at '{CANDIDATE_DB_DIR}'. Build it first."
        )
    return FAISS.load_local(
        CANDIDATE_DB_DIR,
        get_embedder(),
        allow_dangerous_deserialization=True
    )


def _dedupe_docs(docs: List[Document], max_docs: int) -> List[Document]:
    seen = set()
    unique = []
    for d in docs:
        key = (d.metadata.get("source_id"), d.page_content[:200])
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
        if len(unique) >= max_docs:
            break
    return unique


def retrieve_jd_then_candidate(
    jd_vs: FAISS,
    candidate_vs: FAISS,
    task_query: str,
) -> Tuple[List[Document], List[Document]]:
    jd_hits = jd_vs.similarity_search(task_query, k=JD_K)

    cand_hits: List[Document] = []
    for jd_doc in jd_hits:
        cand_hits.extend(candidate_vs.similarity_search(jd_doc.page_content, k=CAND_PER_JD_K))

    cand_hits = _dedupe_docs(cand_hits, MAX_CAND_DOCS)
    return jd_hits, cand_hits


def _format_evidence(title: str, docs: List[Document], max_chars: int = 800) -> str:
    parts = [title]
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source_id") or meta.get("source_file") or "unknown"
        text = d.page_content.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        parts.append(f"\n[{i}] Source: {src}\n{text}")
    return "\n".join(parts)


def _run(prompt_template: PromptTemplate, jd_docs: List[Document], cand_docs: List[Document]) -> str:
    guardrails = (
        "GROUNDING RULES:\n"
        "1) Use JOB DESCRIPTION EVIDENCE to decide what to emphasize.\n"
        "2) Use ONLY CANDIDATE EVIDENCE for claims about the candidate.\n"
        "3) If candidate evidence is missing for a JD requirement, do not invent it. Call it a gap.\n"
        "4) Keep responses concise and specific.\n"
    )

    context = (
        guardrails
        + "\n\n"
        + _format_evidence("JOB DESCRIPTION EVIDENCE:", jd_docs)
        + "\n\n"
        + _format_evidence("CANDIDATE EVIDENCE:", cand_docs)
    )

    system_prompt = (
    "You are a careful interview assistant. "
    "Follow the grounding rules. "
    "Never invent candidate experience."
    )

    user_prompt = prompt_template.format(context=context)
    llm = get_llm()
    return llm.invoke(system=system_prompt, user=user_prompt)


def generate_interview_questions(
    jd_text: str,
    jd_vs: Optional[FAISS] = None,
    candidate_vs: Optional[FAISS] = None
) -> str:
    # Use passed-in stores if available (Streamlit session_state)
    candidate_vs = candidate_vs or load_candidate_vectorstore()
    jd_vs = jd_vs or get_jd_vectorstore_from_text(jd_text)

    task_query = "Extract the most important requirements and generate role-specific interview questions."
    jd_docs, cand_docs = retrieve_jd_then_candidate(jd_vs, candidate_vs, task_query)

    return _run(INTERVIEW_PROMPT, jd_docs, cand_docs)


def generate_cover_letter(
    jd_text: str,
    jd_vs: Optional[FAISS] = None,
    candidate_vs: Optional[FAISS] = None
) -> str:
    candidate_vs = candidate_vs or load_candidate_vectorstore()
    jd_vs = jd_vs or get_jd_vectorstore_from_text(jd_text)

    task_query = "Identify the key responsibilities and draft a tailored cover letter aligned to them."
    jd_docs, cand_docs = retrieve_jd_then_candidate(jd_vs, candidate_vs, task_query)

    return _run(COVER_LETTER_PROMPT, jd_docs, cand_docs)