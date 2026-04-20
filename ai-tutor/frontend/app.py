import streamlit as st
import requests

API_URL = "http://localhost:8000/api/ask"
UPLOAD_URL = "http://localhost:8000/upload/embeddings"
UPLOAD_IMAGE_URL = "http://localhost:8000/upload/image"

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
                    # Store the uploaded filename in session state for the chat
                    st.session_state.current_document = uploaded_file.name
                    st.success(f"Successfully uploaded and processed {uploaded_file.name}!")
                else:
                    st.error(f"Upload failed: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")
    else:
        st.warning("Please select a PDF file first.")

st.header("Upload Images")
uploaded_images = st.file_uploader("Choose image files", type=["png", "jpg", "jpeg", "gif", "webp"], accept_multiple_files=True)

if st.button("Upload Images"):
    if uploaded_images:
        with st.spinner(f"Uploading {len(uploaded_images)} images..."):
            for img in uploaded_images:
                files = {"file": (img.name, img.getvalue(), img.type)}
                try:
                    response = requests.post(UPLOAD_IMAGE_URL, files=files)
                    if response.status_code == 200:
                        metadata = response.json().get("metadata", {})
                        st.success(f"Successfully uploaded {img.name}!")
                        with st.expander(f"View metadata for {img.name}"):
                            st.json(metadata)
                    else:
                        st.error(f"Failed to upload {img.name}: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error for {img.name}: {e}")
    else:
        st.warning("Please select at least one image file first.")

st.divider()

st.header("💬 Chat with AI Tutor")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image"):
            try:
                st.image(message["image"]["filename"], caption=message["image"].get("title", ""))
            except Exception:
                st.info(f"🖼️ [Image reference: {message['image']['filename']}]")

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                current_doc = st.session_state.get("current_document", "document")
                response = requests.post(API_URL, json={"query": prompt, "filename": current_doc})
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    image_data = data.get("image")
                    
                    st.markdown(answer)
                    
                    if image_data:
                        try:
                            # Attempts to load the local file. 
                            # Note: Ensure the frontend and backend share access to this file path!
                            st.image(image_data["filename"], caption=image_data.get("title", ""))
                        except Exception:
                            st.info(f"🖼️ *(Relevant Image: {image_data['filename']} - Please ensure images are in an accessible directory)*")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "image": image_data
                    })
                else:
                    st.error(f"Error from server: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")
