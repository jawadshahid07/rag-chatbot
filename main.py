# main.py
from rag.agent import agent_app
from rag.logger import logger

if __name__ == "__main__":
    logger.info("Starting LangGraph agent...")

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            logger.info("Session ended by user.")
            break

        result = agent_app.invoke({"question": query})
        print("\nAnswer:\n", result["answer"])
