# ============================================
# 📌 SIMPLE CHATBOT UI WITH STREAMLIT
# ============================================

import streamlit as st
import requests

# ✅ FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/chat/"

# ============================================
# 📌 SECTION 1: STREAMLIT PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Falcon-7B Assistant Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Falcon-7B Assistant Chatbot")
st.write("Ask me anything about orders, refunds, and shipping!")

# ============================================
# 📌 SECTION 2: CHAT FUNCTION
# ============================================

def get_chatbot_response(user_query):
    """Send the user query to the FastAPI backend and return the response."""
    try:
        response = requests.post(API_URL, json={"query": user_query})
        if response.status_code == 200:
            return response.json().get("chatbot_response", "❌ No response from chatbot.")
        else:
            return f"❌ API Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"❌ API Request Failed: {str(e)}"

# ============================================
# 📌 SECTION 3: USER INPUT & CHAT INTERFACE
# ============================================

# ✅ Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display previous chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ User input box
user_query = st.chat_input("Type your question here...")

if user_query:
    # ✅ Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # ✅ Get chatbot response
    chatbot_response = get_chatbot_response(user_query)
    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

    with st.chat_message("assistant"):
        st.markdown(chatbot_response)
