# rag/agent.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Union
from langchain_core.runnables import Runnable
from rag.rag_tool import setup_rag_tool
from rag.sql_tool import get_sql_toolkit
from rag.llm import get_llm
from rag.logger import logger
from rag.booking_tool import book_car
from datetime import datetime, timedelta


# Define shared state between steps
class AgentState(TypedDict):
    question: str
    answer: Union[str, None]
    route: Literal["rag", "sql"]
    sql_query: Union[str, None]
    top_model: Union[str, None]  # <-- NEW

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

# Tool: booking

def booking_node(state: AgentState) -> AgentState:
    logger.info("Using Booking tool...")

    extraction_prompt = (
        "Extract the car model, booking date, and customer name from the question below.\n"
        "Return in this format: MODEL | YYYY-MM-DD | CUSTOMER NAME\n"
        "Use today's date if the date is unclear.\n"
        "Use 'Anonymous' if the name is not provided.\n\n"
        f"Question: {state['question']}"
    )

    raw_output = get_llm().invoke(extraction_prompt).strip()
    logger.info(f"Parsed booking info: {raw_output}")

    try:
        model, date, customer = [s.strip() for s in raw_output.split("|")]
    except Exception:
        return {
            "question": state["question"],
            "answer": "❌ Could not parse booking info. Please include model, date, and (optionally) name.",
            "route": "booking"
        }

    confirmation = book_car(model, date, customer)
    return {
        "question": state["question"],
        "answer": confirmation,
        "route": "booking"
    }
    logger.info("Using Weather-based booking...")

    prompt = (
        "From the following user query, extract:\n"
        "MODEL | CITY\n\n"
        f"Question: {state['question']}"
    )

    try:
        raw = get_llm().invoke(prompt).strip()
        model, city = [x.strip() for x in raw.split("|")]
    except:
        return {
            "question": state["question"],
            "answer": "❌ Could not parse model and city.",
            "route": "weather_booking"
        }

    forecast = get_weather_forecast(city)
    logger.info(f"Forecast for {city} tomorrow: {forecast}")

    if "rain" in forecast.lower():
        tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        confirmation = book_car(model, tomorrow)
        answer = f"☔ It will rain tomorrow in {city}. {confirmation}"
    else:
        answer = f"☀ No rain forecast in {city}. No booking made."

    return {
        "question": state["question"],
        "answer": answer,
        "route": "weather_booking"
    }



# Decision router
def decide_route(state: AgentState) -> Literal["rag", "sql"]:
    q = state["question"].lower()

    # Basic rule: if asking about "how many", "sold", "last week", etc., go to SQL
    if any(kw in q for kw in ["sold", "how many", "last week", "yesterday", "total sales", "most sold"]):
        logger.info("Routing to SQL tool")
        return "sql"
    if any(kw in q for kw in ["book", "booking", "reserve", "schedule"]):
        logger.info("Routing to Booking tool")
        return "booking"
    if "book" in q and "rain" in q:
        logger.info("Routing to weather_booking")
        return "weather_booking"


    logger.info("Routing to RAG tool")
    return "rag"


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("rag", rag_node)
graph.add_node("generate_sql", generate_sql_query)
graph.add_node("sql", sql_node)
graph.add_node("router", lambda x: x)  # Pass-through node to run decision logic
graph.add_node("booking", booking_node)
graph.add_node("weather_booking", weather_booking_node)
graph.add_edge("weather_booking", END)


graph.add_conditional_edges("router", decide_route, {
    "rag": "rag",
    "sql": "generate_sql",
    "booking": "booking",
    "weather_booking": "weather_booking"
})
graph.set_entry_point("router")
graph.add_edge("generate_sql", "sql")
graph.add_edge("sql", END)
graph.add_edge("rag", END)
graph.add_edge("booking", END)

# Compile it
agent_app: Runnable = graph.compile()
