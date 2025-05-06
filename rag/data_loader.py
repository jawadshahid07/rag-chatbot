import json
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

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

def load_pdfs_from_folder(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(full_path)
            pdf_docs = loader.load()
            all_docs.extend(pdf_docs)
    return all_docs
