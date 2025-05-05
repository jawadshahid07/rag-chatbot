from rag.data_loader import load_json_documents, chunk_documents
from rag.embedder import create_vectorstore
from rag.retriever import get_retriever
from rag.generator import generate_with_llama
from rag.rag_pipeline import RAGChatbot

if __name__ == "__main__":
    # Load & chunk data
    qa_docs = load_json_documents("data/qa_data.json")
    spec_docs = load_json_documents("data/car_specs.json")
    all_docs = chunk_documents(qa_docs + spec_docs)

    # Vector store
    vs = create_vectorstore(all_docs)
    retriever = get_retriever(vs)

    # RAG pipeline
    chatbot = RAGChatbot(retriever, generate_with_llama)

    print("Automobile Assistant Ready!")
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break
        answer = chatbot.ask(query)
        print("\nAnswer:\n", answer)
