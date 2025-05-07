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
    route: Union[str, None]
    sql_query: Union[str, None]
    tool_used: Union[str, None]  # NEW

def finalize_response_node(state: AgentState) -> AgentState:
    tool = state.get("tool_used") or "unknown"
    question = state.get("question") or ""
    raw_output = state.get("answer")

    # Optional formatter: unpacks SQL-like tuple values e.g. [(100,)] → "100"
    def format_raw_output(output):
        if isinstance(output, list) and len(output) == 1 and isinstance(output[0], tuple):
            return ", ".join(str(i) for i in output[0])
        return str(output)

    formatted_output = format_raw_output(raw_output)

    logger.info(f"Finalization Input — Tool: {tool}, Question: {question}, Output: {formatted_output}")

    prompt = f"""You are a helpful assistant. Given a user's question and the raw output from a tool, rephrase the output into a concise, human-friendly answer.

DO NOT make assumptions or add any new facts. Only rephrase or clarify the tool output.

Examples:
Tool: sql
Question: What was the most sold car?
Output: [('Ford F-150 2019',)]
Answer: The most sold car is the Ford F-150 2019.

Tool: sql
Question: How many sales have been made in total?
Output: [(100,)]
Answer: A total of 100 sales have been made.

Tool: booking
Question: I want to book a Tesla for tomorrow
Output: ✅ Booking confirmed for Tesla Model Y on 2025-05-08 for Anonymous.
Answer: ✅ Booking confirmed for Tesla Model Y on 2025-05-08 for Anonymous.

Tool: rag
Question: What safety features does the 2021 Honda Civic have?
Output: The 2021 Honda Civic includes lane keep assist, adaptive cruise control, and blind spot monitoring.
Answer: The 2021 Honda Civic includes lane keep assist, adaptive cruise control, and blind spot monitoring.

---

Now complete the following:

Tool: {tool}
Question: {question}
Output: {formatted_output}
Answer:"""

    try:
        final_answer = get_llm().invoke(prompt).strip()
    except Exception as e:
        final_answer = f"❌ Could not finalize response. Tool output was: {formatted_output}"
        logger.error(f"Finalization LLM error: {e}")

    logger.info(f"Finalized response: {final_answer}")

    return {
        **state,
        "answer": final_answer,
        "route": "end"
    }


def main_agent_node(state: AgentState) -> AgentState:
    if state.get("tool_used"):
        logger.info(f"Tool '{state['tool_used']}' used. Ending.")
        return {**state, "route": "end"}

    question = state["question"]

    tool_descriptions = {
        "rag": "Use this to answer questions about car specifications, manuals, and features.",
        "sql": "Use this to answer statistical questions about car sales, booking counts, and trends.",
        "booking": "Use this to make a booking of a car for a customer."
    }

    prompt = (
        "You are an intelligent agent deciding which tool to use to answer a question.\n\n"
        "Available tools:\n"
        + "\n".join(f"- {tool}: {desc}" for tool, desc in tool_descriptions.items()) +
        "\n\nInstruction: Based on the question below, choose the most appropriate tool.\n"
        "Respond only with:\nAction: <tool_name>\n\n"
        f"Question: {question}"
    )

    raw_response = get_llm().invoke(prompt).strip()
    logger.info(f"Router LLM response: {raw_response}")

    # Extract tool name
    try:
        tool = raw_response.split("Action:")[1].strip().lower()
        if tool not in tool_descriptions:
            raise ValueError("Invalid tool name")
    except Exception:
        tool = "rag"  # fallback

    logger.info(f"Routing to {tool} tool")
    return {**state, "route": tool}



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
    return {
        "question": state["question"],
        "answer": answer,
        "route": "rag",
        "tool_used": "rag"
    }


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
            "sql_query": sql,
            "route": "sql",
            "tool_used": "sql"
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
            "route": "booking",
            "tool_used": "booking"
        }

    confirmation = book_car(model, date, customer)
    return {
        "question": state["question"],
        "answer": confirmation,
        "route": "booking",
        "tool_used": "booking"
    }


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("main", main_agent_node)
graph.add_node("rag", rag_node)
graph.add_node("generate_sql", generate_sql_query)
graph.add_node("sql", sql_node)
graph.add_node("booking", booking_node)
graph.add_node("finalize", finalize_response_node)


graph.set_entry_point("main")

graph.add_conditional_edges("main", lambda state: state["route"], {
    "rag": "rag",
    "sql": "generate_sql",
    "booking": "booking",
    "end": END
})



graph.add_edge("generate_sql", "sql")

# All tools return to "main"
graph.add_edge("rag", "finalize")
graph.add_edge("sql", "finalize")
graph.add_edge("booking", "finalize")


# Compile it
agent_app: Runnable = graph.compile()
