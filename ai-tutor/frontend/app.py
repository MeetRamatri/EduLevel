import streamlit as st
import requests

API_URL = "http://localhost:8000/api/ask"
UPLOAD_URL = "http://localhost:8000/upload"

st.set_page_config(page_title="AI Tutor", page_icon="🎓")

st.title("🎓 AI Tutor")

st.header("Upload Document")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if st.button("Upload PDF"):
    if uploaded_file is not None:
        with st.spinner("Uploading and processing PDF..."):
            # Prepare the file for the multipart/form-data request
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post(UPLOAD_URL, files=files)
                if response.status_code == 200:
                    st.success(f"Successfully uploaded and processed {uploaded_file.name}!")
                else:
                    st.error(f"Upload failed: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")
    else:
        st.warning("Please select a PDF file first.")
