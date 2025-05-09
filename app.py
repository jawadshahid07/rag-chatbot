# app.py
import streamlit as st
from rag.agent import agent_app
from rag.logger import logger
import requests

st.set_page_config(page_title="🚗 Automobile Assistant", layout="wide")
st.title("🚗 Automobile Assistant (RAG + SQL)")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# Input box for new question
if user_input := st.chat_input("Ask a question about cars, features, or sales..."):
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Thinking..."):
        try:
            response = requests.post("http://localhost:5678/webhook/automobile-agent", json={"question": user_input})
            response_data = response.json()
            response = response_data.get("answer", "Sorry, no answer.")
        except Exception as e:
            logger.error(f"Agent error: {e}")
            response = "An error occurred while processing your question."

    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append(("assistant", response))
