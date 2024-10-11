import streamlit as st
# from langchain import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_ollama import ChatOllama

# The UI Part
st.title("👨‍💻 Let's chat about the meeting!")
uploaded_file = st.file_uploader("Upload a .txt file of the meeting transcript here", type=["txt"])
load_button = st.button("Load meeting transcript.", type="primary")
# question = st.text_area(
#     "Now ask a question about the document!",
#     placeholder="Can you give me a short summary?",
#     disabled=not uploaded_file,
# )
prompt = None


# Load VectorDB
if load_button: 
    if uploaded_file:
        text = uploaded_file.getvalue()
        print(type(text))
        print(type(uploaded_file.read().decode()))
        # docs = Document(page_content=text.decode('utf-8'))
        # Change UI
        st.text("You did it!")
        transcript = None
        load_button = None
        prompt = st.text_area("Please enter what you want to know from the meeting.")
    else:
        st.write("MUST UPLOAD FILE!")

if uploaded_file and prompt:
    st.write("Nice")