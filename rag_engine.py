import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def load_and_index_pdfs(pdf_paths: list[str]):
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="llama-3.3-70b-versatile")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def build_qa_chain(vectorstore):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.2
    )

    prompt = PromptTemplate.from_template("""You are a research assistant helping synthesize findings across multiple academic papers.

Use the context below — drawn from multiple papers — to answer the question.
Always mention which paper(s) each point comes from using the source name in brackets like [paper.pdf].
If papers agree, highlight the consensus. If they disagree, explicitly state the disagreement.
If you don't know, say so — do not hallucinate.

Context:
{context}

Question:
{question}

Answer:""")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]:\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


def query_papers(chain_dict, question: str):
    chain = chain_dict["chain"]
    retriever = chain_dict["retriever"]

    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    sources = list(set(
        doc.metadata.get("source", "unknown") for doc in source_docs
    ))
    return answer, sources