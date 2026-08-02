import streamlit as st
import requests

API_URL = "https://rag-document-qa-api-c7ul.onrender.com"

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Q&A")
st.write("Upload a PDF and ask questions about its content.")

# --- Upload section ---
st.header("1. Upload a document")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Process document"):
        with st.spinner("Extracting, chunking, and embedding..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_URL}/upload", files=files,timeout=60)
    try:
        response = requests.post(f"{API_URL}/upload", files=files, timeout=60)
        if response.status_code == 200:
            data = response.json()
            st.success(f"Indexed {data['chunks_created']} chunks from {data['filename']}")
            st.session_state["document_ready"] = True
        else:
            st.error(f"Upload failed. Status code: {response.status_code}")
            st.code(response.text)
    except requests.exceptions.RequestException as e:
     st.error(f"Connection error: {e}")

# --- Ask section ---
st.header("2. Ask a question")
question = st.text_input("Your question")

if st.button("Get answer"):
    if not question:
        st.warning("Please type a question first.")
    else:
        with st.spinner("Retrieving relevant context and generating answer..."):
            response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)

        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                st.warning(data["error"])
            else:
                st.subheader("Answer")
                st.write(data["answer"])
                st.caption(f"Based on {data['sources_used']} retrieved chunks")
        else:
            st.error("Something went wrong. Check the backend logs.")