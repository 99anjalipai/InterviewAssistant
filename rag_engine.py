from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama

from prompts import INTERVIEW_PROMPT, COVER_LETTER_PROMPT


def chunk_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])


def get_vectorstore_from_text(text: str):
    chunks = chunk_text(text)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedder)
    return vectorstore


def build_qa_chain(vectorstore, prompt_template: PromptTemplate):
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="gemma:2b")  

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    return chain


def generate_interview_questions(jd_text: str):
    vectorstore = get_vectorstore_from_text(jd_text)
    chain = build_qa_chain(vectorstore, INTERVIEW_PROMPT)
    return chain.run(jd_text)


def generate_cover_letter(jd_text: str):
    vectorstore = get_vectorstore_from_text(jd_text)
    chain = build_qa_chain(vectorstore, COVER_LETTER_PROMPT)
    return chain.run(jd_text)
