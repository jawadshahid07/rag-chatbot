# app.py
import streamlit as st
import requests

st.set_page_config(page_title="🚗 Automobile Assistant", layout="wide")
st.title("🚗 Automobile Assistant (RAG + SQL via n8n)")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# Chat input
if user_input := st.chat_input("Ask a question about cars, features, or sales..."):
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                "https://n8n.hytgenx.co/webhook/chat",
                json={"question": user_input},
                timeout=15
            )
            response_data = response.json()
            answer = response_data.get("output", "Sorry, no answer was returned.")
        except Exception as e:
            answer = f"❌ Error talking to the assistant: {e}"

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append(("assistant", answer))
