from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
# import chromadb
# from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import ollama

# The UI Part
st.title("👨‍💻 Let's chat about the meeting!")
transcript = st.file_uploader("Upload a .txt file of the meeting transcript here", type=["txt"])
load_button = st.button("Load meeting transcript.", type="primary")
prompt = None

# Load VectorDB
if load_button: 
    if transcript is not None:
        text = transcript.getvalue()
        docs = Document(page_content=text.decode('utf-8'))
        st.write(text)
        # Change UI
        st.text("You did it!")
        transcript = None
        load_button = None
        prompt = st.text_area("Please enter what you want to know from the meeting.")
