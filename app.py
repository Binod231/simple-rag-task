import streamlit as st
import os
from pinecone import Pinecone
# from pinecone.specs import ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # New, correct import
from langchain_pinecone import PineconeVectorStore      # New, correct import
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Import configuration variables
from config import (
    PINECONE_API_KEY,
    PINECONE_API_ENV,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL
)

# Set up the Streamlit page
st.set_page_config(page_title="AI Research Explorer", layout="wide")
st.title("AI Research Explorer")
st.write("Have questions about Artificial Intelligence? Ask anything about foundational AI papers on topics like Transformers, BERT, and LLMs. Get clear answers, sourced directly from the research.")

# Add a section for example questions based on your documents
with st.expander("🤔 Need inspiration? Try asking..."):
    st.info("""
    - What is the main idea of the 'Attention Is All You Need' paper?
    - In simple terms, what is Chain-of-Thought Prompting?
    - What problem does the FlashAttention paper solve?
    - How is DistilBERT different from the original BERT?
    - Summarize the LLaMA paper.
    """)

# --- 1. Load and Process Documents ---
@st.cache_resource
def load_and_process_documents():
    """
    Loads documents from the 'data' directory and splits them into chunks.
    """
    loader = DirectoryLoader('./data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts

texts = load_and_process_documents()

# --- 2. Create Embeddings and Store in Pinecone ---
@st.cache_resource
def setup_vector_database(texts):
    """
    Creates embeddings and initializes the Pinecone vector store using modern syntax.
    """
    # Use the new, correct HuggingFaceEmbeddings class
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Initialize the Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if the index exists and create it if it doesn't
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        # This is the NEW, correct way
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric='cosine',
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": PINECONE_API_ENV
                }
            }
        )
    
    # Use the new PineconeVectorStore class to create the index from documents
    docsearch = PineconeVectorStore.from_documents(
        texts,
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    return docsearch

docsearch = setup_vector_database(texts)

# --- 3. Create the RAG Chain ---
@st.cache_resource
def create_rag_chain(_retriever): # <-- UNDERSCORE ADDED HERE
    """
    Creates the RetrievalQA chain with source document tracking.
    """
    llm = Ollama(model=LLM_MODEL)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_retriever, # <-- UNDERSCORE ADDED HERE
        return_source_documents=True
    )
    return qa

# The retriever is now created from the docsearch object
retriever = docsearch.as_retriever()
qa_chain = create_rag_chain(retriever) # No change needed here

# The retriever is now created from the docsearch object
retriever = docsearch.as_retriever()
qa_chain = create_rag_chain(retriever)

# --- 4. User Interface ---
st.header("Ask a Question")
query = st.text_input("Enter your question about the research papers:")

if st.button("Get Answer"):
    if query:
        with st.spinner(f"Finding the answer with your local {LLM_MODEL}..."):
            try:
                result = qa_chain.invoke({"query": query}) # Use .invoke() for modern LangChain
                
                st.success("Here is the answer:")
                st.write(result["result"])

                st.subheader("Sources Used to Generate the Answer:")
                for doc in result["source_documents"]:
                    # Clean up the source path for better display
                    source_name = doc.metadata.get('source', 'N/A').split('/')[-1].split('\\')[-1]
                    with st.expander(f"Source: {source_name}"):
                        st.write(doc.page_content)

            except Exception as e:
                st.error(f"An error occurred. Make sure the Ollama application is running. Error: {e}")
    else:
        st.warning("Please enter a question.")