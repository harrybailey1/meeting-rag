import streamlit as st
from rag import RAG

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# The UI Part
st.title("👨‍💻 Let's chat about the meeting!")
uploaded_file = st.file_uploader("Upload a .txt file of the meeting transcript here", type=["txt"])
load_button = st.button("Load meeting transcript", type="primary")
question = st.text_area(
    "Please enter what you want to know from the meeting.",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)
warning_text = st.empty()

# RAG placeholder
rag = None

# Load VectorDB
if load_button: 
    if uploaded_file:
        # Update IU
        warning_text.empty()
        # text = uploaded_file.getvalue()
        text = uploaded_file.read().decode()
        # Set up RAG instance
        rag = RAG(text)
    else:
        # Update UI
        warning_text.text("MUST UPLOAD FILE!")

# Process question
if uploaded_file and question:
    if rag:
        # Update UI
        warning_text.empty()
        # Run process query on RAG instance
        stream = rag.process_query(question)
        # Write out stream on site
        st.write_stream(stream)
    else:
        # Update UI
        warning_text.text("Please load the transcript first")