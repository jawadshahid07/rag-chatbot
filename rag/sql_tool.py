# rag/sql_tool.py
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit 
from rag.llm import get_llm  # Reuse your existing Ollama wrapper

def get_sql_toolkit():
    llm = get_llm()
    db = SQLDatabase.from_uri("sqlite:///db/car_sales.db")  # Local file
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit

if __name__ == "__main__":
    toolkit = get_sql_toolkit()
    query_tool = toolkit.get_tools()[0]

    # Real SQL for testing
    sql = """
    SELECT model, SUM(quantity) AS total_sold
    FROM car_sales
    WHERE sale_date >= date('now', '-7 days')
    GROUP BY model
    ORDER BY total_sold DESC
    LIMIT 1
    """
    result = query_tool.invoke(sql)
    print(result)
