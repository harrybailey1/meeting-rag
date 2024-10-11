import streamlit as st
from rag import RAG

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# The UI Part
st.title("👨‍💻 Let's chat about the meeting!")
uploaded_file = st.file_uploader("Upload a .txt file of the meeting transcript here", type=["txt"])
load_button = st.button("Load meeting transcript.", type="primary")
# Placeholders for later
prompt = None
rag = None
# question = st.text_area(
#     "Now ask a question about the document!",
#     placeholder="Can you give me a short summary?",
#     disabled=not uploaded_file,
# )


# Load VectorDB
if load_button: 
    if uploaded_file:
        text = uploaded_file.getvalue()
        # Set up RAG instance
        rag = RAG(text)
        # Change UI
        prompt = st.text_area("Please enter what you want to know from the meeting.")
    else:
        st.write("MUST UPLOAD FILE!")

# Process question
if uploaded_file and prompt:
    # Run process query on RAG instance
    stream = RAG.process_query(prompt)
    # Write out stream on site
    print(stream)
    st.write_stream(stream)