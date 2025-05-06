from rag.logger import logger

class RAGChatbot:
    def __init__(self, retriever, generator_func):
        self.retriever = retriever
        self.generator = generator_func

    def ask(self, query):
        logger.info(f"User query: {query}")

        # Step 1: Retrieve documents
        docs = self.retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents from vector store.")

        for i, doc in enumerate(docs):
            logger.info(f"Doc {i + 1}:\n{doc.page_content[:400]}...\n")

        # Step 2: Create final prompt
        context = "\n".join(doc.page_content for doc in docs)
        prompt = (
            "Use the following information to answer the question. "
            "If the context is unclear or incomplete, state that explicitly.\n\n"
            f"{context}\n\nQuestion: {query}\nAnswer:"
        )

        logger.info("Final prompt constructed for LLM.")

        # Step 3: Generate response
        try:
            result = self.generator(prompt)
            logger.info("LLM generation successful.")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result = "Error: Unable to generate a response."

        return result
