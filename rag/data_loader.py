import json
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_json_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    docs = []
    for item in items:
        content = ""
        if "question" in item and "answer" in item:
            content = f"Q: {item['question']}\nA: {item['answer']}"
        elif "make" in item:
            content = f"{item['year']} {item['make']} {item['model']} - {item['summary']}\nFeatures: {', '.join(item['features'])}"
        docs.append(Document(page_content=content))
    return docs

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
