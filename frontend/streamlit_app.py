import streamlit as st
import requests
#import re

st.set_page_config(page_title="Dell Laptop RAG Assistant", page_icon="💻")
st.title("💻 Dell Laptop RAG Assistant")

with st.form("ask_form"):
    query = st.text_input("Enter your question:")
    submit = st.form_submit_button("Ask Assistant")

if submit:
    if query.strip():
        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={"question": query},
                timeout=60
            )
            data = response.json()
            st.markdown("### 🤖 Assistant Response")
            st.write(data["answer"])
        except Exception as e:
            st.error("❌ Failed to connect to backend")
            st.text(str(e))
    else:
        st.warning("⚠️ Please enter a question before asking.")