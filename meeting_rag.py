# import sys
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
# # import chromadb
# # from chromadb.utils import embedding_functions
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddings,
# )
# from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores import Chroma
# from langchain.tools.retriever import create_retriever_tool
# from langchain import hub
# from langchain.agents import AgentExecutor
# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper

# The UI Part
st.title("👨‍💻 Let's chat about the meeting!")
transcript = st.file_uploader("Upload a .txt file of the meeting transcript here")
prompt = st.text_area("Please enter what you want to know from the meeting.")

# # Load VectorDB
# if st.sidebar.button("Load meeting transcript into Vector DB if loading the page for the first time.", type="primary"): 
#     loader = TextLoader("hansardFeb2024.txt")
#     docs = loader.load()
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     text_splitter = RecursiveCharacterTextSplitter(
#        # Set chunk size, just to show.
#         chunk_size=750,
#         chunk_overlap=50,
#         length_function=len,
#         is_separator_regex=False,
#     )
    
#     documents = text_splitter.split_documents(docs)
#     vectorstore = Chroma.from_documents(documents, embeddings)
#     retriever = vectorstore.as_retriever()
#     # tool = create_retriever_tool(
#         # retriever,
#         # "Search_Hansard",
#         # "Searches and returns Hansard data.",
#     # )
#     # retriever_tool = [tool]
#     retriever_tool = create_retriever_tool(
#         retriever,
#         "handsard_search",
#         "Search for information about Handsard. For any questions about Handsard, you must use this tool!",
#     )
#     tools = [retriever_tool]
#     prompt_template = hub.pull("hwchase17/openai-tools-agent")
#     # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=st.secrets["api_key"])
#     llm = ChatOpenAI(temperature=0, api_key=st.secrets["api_key"])
#     agent = create_openai_tools_agent(llm, tool, prompt_template)
#     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#     st.write("Agent Ready to run.")

# if st.button("Submit to DJ Arvee", type="primary"):
#     # Get the prompt to use - you can modify this! 
#     st.write(agent_executor.invoke({"input": prompt}))



# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import streamlit as st
# import ollama
# import chromadb
# from chromadb.utils import embedding_functions
# # from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

# CHROMA_DATA_PATH = "chroma_data/"
# EMBED_MODEL = "all-MiniLM-L6-v2"
# COLLECTION_NAME = "demo_docs"

# client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
# st.write("after ChromaDB client create")

# embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
#      model_name=EMBED_MODEL
#  )

# st.write("after embedding function create")

# # collection = client.create_collection(
# collection = client.get_or_create_collection(
#      name=COLLECTION_NAME,
#      embedding_function=embedding_func,
#      metadata={"hnsw:space": "cosine"},
#  )

# with open("hansard-utf8.txt") as f:
#     hansard = f.read()

# text_splitter = RecursiveCharacterTextSplitter(
# #     Set a really small chunk size, just to show.
#     chunk_size=500,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )

# texts = text_splitter.create_documents([hansard])
# st.write(texts[0])
# # print(texts[1])
# # print(texts[2])
# # print(texts[3])

# documents = text_splitter.split_text(hansard)[:len(texts)]
# st.write(documents)

# collection.add(
#      documents=documents,
#      ids=[f"id{i}" for i in range(len(documents))],
# #     metadatas=[{"genre": g} for g in genres]
# )

# # number of rows
# st.write(len(collection.get()['documents']))

# prompt = ("What were the key points provided by Ms Jenkins relating to underpayment in the department?")

# query_results = collection.query(
#      query_texts=[prompt],
#      # include=["documents", "embeddings"],
#      include=["documents"],
#      n_results=100,
#  )

# # print(query_results["embeddings"])
# # st.write(query_results["documents"])

# augment_query = str(query_results["documents"])
# st.write(augment_query)

# response = ollama.chat(
#     model='llama2',
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a friendly assistant."
#         },
#         {
#             "role": "user",
#             "content": augment_query + " Prompt: " + prompt
#         },
#     ],
# )

# st.write(response['message']['content'])