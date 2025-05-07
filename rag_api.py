# rag_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag.data_loader import load_json_documents, chunk_documents, load_pdfs_from_folder
from rag.vectorstore import build_faiss_vectorstore
from rag.llm import get_llm
from rag.rag_chain import build_rag_chain

app = FastAPI()

# Preload components
qa_docs = load_json_documents("data/qa_data.json")
spec_docs = load_json_documents("data/car_specs.json")
pdf_docs = load_pdfs_from_folder("data/pdfs")
all_docs = chunk_documents(qa_docs + spec_docs + pdf_docs)

vectorstore = build_faiss_vectorstore(all_docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = get_llm()
rag_chain = build_rag_chain(llm, retriever)

class Query(BaseModel):
    query: str

@app.post("/rag")
def ask(query: Query):
    result = rag_chain.invoke(query.query)
    return {"result": result["result"]}
