def get_retriever(vectorstore, k=5):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever
