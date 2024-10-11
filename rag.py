from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain import hub
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAG:
    def __init__(self, file_text, model="gpt-4o-mini", chunk_size=500, chunk_overlap=50, separator="\n"):
        self.file_text = file_text
        # Create text splitter (default separator is \n\n)
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap, 
                                                   separator=separator)
        # Create OpenAI embeddings for the documents
        self.embedding = OpenAIEmbeddings()
        # Initialise OpenAI chat model with regular params
        self.model = ChatOpenAI(
            model=model,
            temperature=0.7,
            streaming=True
        )
        # Initialise retriever
        self.retriever = self.get_retriever()
        # Initialise prompt template
        self.prompt_template = hub.pull("langchain-ai/retrieval-qa-chat")
        # Initialise chain
        self.chain = self.get_chain()
    
    # Function for combining multiple retrieved documents into text
    def format_retrieved_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # Return retriever from document
    def get_retriever(self):
        doc = Document(page_content=self.file_text, metadata={"source": "local"})
        # Split the document
        texts = self.text_splitter.split_documents(doc)
        # Create a FAISS vector store with the embeddings
        vector_store = FAISS.from_documents(texts, self.embedding)
        # Create a retriever to search for documents
        retriever = vector_store.as_retriever()
        return retriever

    # Return a langchain RAG chain
    def get_chain(self):
        chain = (
            {"context": self.retriever | self.format_retrieved_docs, "input": RunnablePassthrough()}
            | self.prompt
            | self.model
            # | StrOutputParser()
        )
        return chain

    # Return the result of running chain on query
    def process_query(self, query):
        return self.chain.invoke(query)