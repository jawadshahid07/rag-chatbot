# rag/agent.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Union
from langchain_core.runnables import Runnable
from rag.rag_tool import setup_rag_tool
from rag.sql_tool import get_sql_toolkit
from rag.llm import get_llm
from rag.logger import logger


# Define shared state between steps
class AgentState(TypedDict):
    question: str
    answer: Union[str, None]
    route: Literal["rag", "sql"]
    sql_query: Union[str, None]

def generate_sql_query(state: AgentState) -> AgentState:
    prompt = (
        "You are an expert SQL generator for a SQLite database.\n"
        "The table is named `car_sales` and has the following columns:\n"
        "  - `model` (TEXT)\n"
        "  - `quantity` (INTEGER)\n"
        "  - `sale_date` (TEXT in YYYY-MM-DD format)\n\n"
        "Given a natural language question, output ONLY the raw SQL query needed to answer it.\n"
        "Use proper SQLite syntax — for example, use DATE('now', '-1 day') to refer to yesterday.\n"
        "Do not include explanations, markdown formatting, or anything else — just the SQL.\n\n"
        f"Question: {state['question']}\nSQL:"
    )

    sql = get_llm().invoke(prompt).strip()
    logger.info(f"Generated SQL: {sql}")
    return {
        "question": state["question"],
        "sql_query": sql,
        "route": "sql",
        "answer": None
    }


# Tool: RAG
rag_tool = setup_rag_tool()

def rag_node(state: AgentState) -> AgentState:
    logger.info("Using RAG tool...")
    answer = rag_tool.invoke(state["question"])
    return {"question": state["question"], "answer": answer, "route": "rag"}


# Tool: SQL
sql_toolkit = get_sql_toolkit()
sql_query_tool = sql_toolkit.get_tools()[0]  # This is QuerySQLDataBaseTool

def sql_node(state: AgentState) -> AgentState:
    logger.info("Using SQL tool...")
    sql = state["sql_query"]
    try:
        answer = sql_query_tool.invoke(sql)
    except Exception as e:
        answer = f"Error: {e}"
    return {
        "question": state["question"],
        "answer": answer,
        "route": "sql",
        "sql_query": sql
    }


# Decision router
def decide_route(state: AgentState) -> Literal["rag", "sql"]:
    q = state["question"].lower()

    # Basic rule: if asking about "how many", "sold", "last week", etc., go to SQL
    if any(kw in q for kw in ["sold", "how many", "last week", "yesterday", "total sales", "most sold"]):
        logger.info("Routing to SQL tool")
        return "sql"
    logger.info("Routing to RAG tool")
    return "rag"


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("rag", rag_node)
graph.add_node("generate_sql", generate_sql_query)
graph.add_node("sql", sql_node)
graph.add_node("router", lambda x: x)  # Pass-through node to run decision logic

graph.add_conditional_edges("router", decide_route, {
    "rag": "rag",
    "sql": "generate_sql"
})
graph.set_entry_point("router")
graph.add_edge("generate_sql", "sql")
graph.add_edge("sql", END)
graph.add_edge("rag", END)

# Compile it
agent_app: Runnable = graph.compile()
