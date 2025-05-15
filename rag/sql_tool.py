# rag/sql_tool.py
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit 
from rag.llm import get_llm  # Reuse your existing Ollama wrapper

def get_sql_tools():
    llm = get_llm()
    db = SQLDatabase.from_uri("sqlite:///db/car_sales.db")  # Local file
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()