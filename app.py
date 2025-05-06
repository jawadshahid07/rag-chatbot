import streamlit as st
from rag.data_loader import load_json_documents, chunk_documents, load_pdfs_from_folder
from rag.vectorstore import build_faiss_vectorstore
from rag.llm import get_llm
from rag.rag_chain import build_rag_chain
from rag.logger import logger

@st.cache_resource
def setup_rag_chain():
    logger.info("Initializing LangChain RAG pipeline...")
    
    # Load documents
    qa_docs = load_json_documents("data/qa_data.json")
    spec_docs = load_json_documents("data/car_specs.json")
    pdf_docs = load_pdfs_from_folder("data/pdfs")
    all_docs = chunk_documents(qa_docs + spec_docs + pdf_docs)
    
    # Build vectorstore and retriever
    vectorstore = build_faiss_vectorstore(all_docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # LLM and RAG chain
    llm = get_llm()
    return build_rag_chain(llm, retriever)

# UI
st.title("🚗 Automobile Assistant")
user_input = st.text_input("Ask a question about a car:", "")

if user_input:
    rag_chain = setup_rag_chain()
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_input)
        st.markdown("### 📋 Answer:")
        st.write(response["result"])