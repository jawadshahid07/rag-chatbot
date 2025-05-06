from langchain.chains import RetrievalQA
from langchain.schema.runnable import Runnable

def build_rag_chain(llm, retriever) -> Runnable:
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
