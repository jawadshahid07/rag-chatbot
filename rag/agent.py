# rag/agent.py

from langchain.agents import Tool, AgentType, initialize_agent
from rag.llm import get_llm
from rag.rag_chain import build_rag_chain
from rag.data_loader import load_json_documents, load_pdfs_from_folder, chunk_documents
from rag.vectorstore import build_faiss_vectorstore
from rag.sql_tool import get_sql_toolkit
from rag.logger import logger

def get_agent():
    logger.info("Setting up hybrid RAG + SQL agent...")

    # Load and chunk all documents
    qa_docs = load_json_documents("data/qa_data.json")
    spec_docs = load_json_documents("data/car_specs.json")
    pdf_docs = load_pdfs_from_folder("data/pdfs")
    all_docs = chunk_documents(qa_docs + spec_docs + pdf_docs)

    # Setup vectorstore and retriever
    vectorstore = build_faiss_vectorstore(all_docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Setup RAG tool
    rag_chain = build_rag_chain(get_llm(), retriever)
    rag_tool = Tool(
        name="car_manual_qa",
        func=lambda q: rag_chain.invoke(q)["result"],
        description="Use this tool to answer questions about car models, features, and specifications."
    )

    # Setup SQL toolkit
    sql_toolkit = get_sql_toolkit()
    sql_tools = sql_toolkit.get_tools()

    # Combine tools into agent
    tools = [rag_tool] + sql_tools
    agent = initialize_agent(
        tools,
        get_llm(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent
