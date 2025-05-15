# app.py
import streamlit as st
from rag.agent import agent_app
from rag.logger import logger
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="🚗 Automobile Assistant", layout="wide")
st.title("🚗 Automobile Assistant (RAG + SQL + Booking)")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display existing chat history
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# Input box
if user_input := st.chat_input("Ask me about car sales, features, bookings..."):
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Thinking..."):
        try:
            result = agent_app.invoke({
                "messages": [HumanMessage(content=user_input)],
                "user_query": user_input
            })

            final_message = result["messages"][-1]
            response = getattr(final_message, 'content', 'Sorry, no response generated.')
        except Exception as e:
            logger.error(f"Agent error: {e}")
            response = "❌ Something went wrong. Please try again."

    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append(("assistant", response))
