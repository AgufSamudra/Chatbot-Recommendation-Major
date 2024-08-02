import streamlit as st
from major_recomendation import major_recomendation

st.title("Maya Bot")

# Sidebar
with st.sidebar:
    
    new_chat_button = st.button("New Chat", type="primary")
    
    if new_chat_button:
        if len(st.session_state.messages) > 1:
            st.session_state.messages = [st.session_state.messages[0]]


if __name__ == "__main__":
    major_recomendation()

