from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict

from rag.rag_tool import setup_rag_tool
from rag.sql_tool import get_sql_tools
from rag.booking_tool import get_booking_tool
from rag.llm import get_llm
from rag.logger import logger
from rag.prompts import MODEL_SYSTEM_MESSAGE, RESPONSE_FORMAT_SYSTEM_MESSAGE


# ------------------------------
# Agent State with message flow
# ------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str


# ------------------------------
# Main Agent Node (LLM decision)
# ------------------------------
def main_agent_node(state: AgentState) -> AgentState:
    logger.info(f"[main_agent_node] State: {state}")

    llm = get_llm().bind_tools(tools)
    system_prompt = SystemMessage(MODEL_SYSTEM_MESSAGE)

    logger.info(f"[main_agent_node] Invoking LLM with messages: {state['messages']}")
    response = llm.invoke([system_prompt] + state["messages"])

    logger.info(f"[main_agent_node] LLM Response: {response}")

    return {"messages": [response], "user_query": state["user_query"]}


# ------------------------------
# Finalize Response Node
# ------------------------------
def finalize_response_node(state: AgentState) -> AgentState:
    logger.info(f"[finalize_response_node] State: {state}")

    final_message = state["messages"][-1]
    raw_output = getattr(final_message, 'content', '')

    prompt = f"User Query: {state['user_query']}\nRaw Output: {raw_output}\nAnswer:"

    try:
        llm = get_llm()
        system_prompt = SystemMessage(RESPONSE_FORMAT_SYSTEM_MESSAGE)
        logger.info(f"[finalize_response_node] Invoking LLM to finalize: {prompt}")
        final_answer = llm.invoke([system_prompt, HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        final_answer = f"❌ Could not finalize response. Original output: {raw_output}"
        logger.error(f"[finalize_response_node] Finalization error: {e}")

    logger.info(f"[finalize_response_node] Final Answer: {final_answer}")

    return {"messages": [HumanMessage(content=final_answer)], "user_query": state["user_query"]}


# ------------------------------
# ToolNode Setup
# ------------------------------
rag_tool = setup_rag_tool()
sql_tools = get_sql_tools()  # sql_query, sql_table_info, sql_list_tables
booking_tool = get_booking_tool()

tools = sql_tools + [rag_tool, booking_tool]
tool_node = ToolNode(tools)


# ------------------------------
# LangGraph Build
# ------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", main_agent_node)
graph.add_node("tools", tool_node)
graph.add_node("finalize", finalize_response_node)

graph.set_entry_point("agent")

# Control flow: decide whether to call tools again or finalize
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    has_tool_call = getattr(last_message, 'tool_calls', None)
    if has_tool_call:
        logger.info("[should_continue] Tool call detected, continue to tools.")
        return "continue"
    logger.info("[should_continue] No tool call, go to finalize.")
    return "finalize"

graph.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "finalize": "finalize"
})

graph.add_edge("tools", "agent")
graph.add_edge("finalize", END)

agent_app: Runnable = graph.compile()

