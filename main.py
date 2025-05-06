from rag.data_loader import load_json_documents, chunk_documents, load_pdfs_from_folder
from rag.embedder import create_vectorstore
from rag.retriever import get_retriever
from rag.generator import generate_with_llama
from rag.rag_pipeline import RAGChatbot
from rag.logger import logger

if __name__ == "__main__":
    logger.info("Loading documents...")

    # Load all document types
    qa_docs = load_json_documents("data/qa_data.json")
    spec_docs = load_json_documents("data/car_specs.json")
    pdf_docs = load_pdfs_from_folder("data/pdfs")

    # Combine and chunk
    all_docs = chunk_documents(qa_docs + spec_docs + pdf_docs)
    logger.info(f"Total chunks after splitting: {len(all_docs)}")

    # Vectorstore and retriever
    vs = create_vectorstore(all_docs)
    retriever = get_retriever(vs)

    # Chatbot pipeline
    chatbot = RAGChatbot(retriever, generate_with_llama)

    logger.info("Automobile Assistant Ready!")

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            logger.info("Session ended by user.")
            break
        answer = chatbot.ask(query)
        print("\nAnswer:\n", answer)
