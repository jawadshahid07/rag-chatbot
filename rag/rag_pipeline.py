class RAGChatbot:
    def __init__(self, retriever, generator_func):
        self.retriever = retriever
        self.generator = generator_func

    def ask(self, query):
        docs = self.retriever.get_relevant_documents(query)
        context = "\n".join(doc.page_content for doc in docs)
        prompt = f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        return self.generator(prompt)
