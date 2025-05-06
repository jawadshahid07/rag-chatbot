# rag/rag_tool.py

from rag.data_loader import load_json_documents, load_pdfs_from_folder, chunk_documents
from rag.vectorstore import build_faiss_vectorstore
from rag.llm import get_llm
from rag.rag_chain import build_rag_chain
from rag.logger import logger
from langchain.agents import Tool

def setup_rag_tool():
    logger.info("Loading and processing documents...")

    qa_docs = load_json_documents("data/qa_data.json")
    spec_docs = load_json_documents("data/car_specs.json")
    pdf_docs = load_pdfs_from_folder("data/pdfs")
    all_docs = chunk_documents(qa_docs + spec_docs + pdf_docs)

    logger.info(f"Total chunks after splitting: {len(all_docs)}")

    vectorstore = build_faiss_vectorstore(all_docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = get_llm()
    rag_chain = build_rag_chain(llm, retriever)

    # Optional: log top retrieved docs inside rag_chain.ask if needed

    tool = Tool(
        name="car_manual_qa",
        func=lambda q: rag_chain.invoke(q)["result"],
        description=(
            "Use this tool to answer questions about automobile specifications, features, or manuals. "
            "Input should be a plain natural language question. "
            "Example: 'Tell me about Toyota Corolla 2020'"
        )
    )

    return tool