import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import traceback
from config import LLM_MODEL

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Financial Document Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. THE CORE LOGIC: A Single Cached Function ---
@st.cache_resource
def get_qa_chain_for_community(_community):
    """
    Loads data, builds the vector database, and creates the RAG chain.
    This entire setup is cached. The cache is keyed by the _community name.
    """
    print(f"--- CACHE MISS: Running full setup for community: '{_community}' ---")
    try:
        loader = DirectoryLoader(
            f'./data/{_community}',
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()
        if not documents:
            st.error(f"No PDF documents found for {_community}. Please check the data folder.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        persist_directory = f'db_{_community}'
        docsearch = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
        retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 5})

        llm = Ollama(model=LLM_MODEL)
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Please provide the source document and page number for each piece of information you use.

        Context: {context}
        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return qa_chain
    except Exception as e:
        st.error(f"An error occurred during setup for {_community}: {e}")
        print(traceback.format_exc())
        return None

# --- 3. Sidebar and Community Selection ---
st.sidebar.title("Community Selection")
try:
    communities = sorted([d for d in os.listdir('./data') if os.path.isdir(os.path.join('./data', d))])
    if not communities:
        st.sidebar.error("No communities found. Please create subdirectories in 'data'.")
        st.stop()
    
    # Use session_state to track the selected community
    if 'community' not in st.session_state:
        st.session_state.community = communities[0]

    # The selectbox now updates the session_state
    def on_community_change():
        # This function will be called when the selectbox value changes
        # We clear the cache here to force a reload of the data
        get_qa_chain_for_community.clear()

    community = st.sidebar.selectbox(
        "Select a company to query:",
        communities,
        key='community_selector', # Add a key for stability
        on_change=on_community_change # This is the crucial fix
    )

except FileNotFoundError:
    st.sidebar.error("The 'data' directory was not found. Please create it.")
    st.stop()

# --- 4. Main Application Logic ---
st.title("Financial Document Explorer")
st.header(f"Ask a Question about {community}'s 10-K Reports")

with st.spinner(f"Setting up the knowledge base for {community}... This may take a moment."):
    qa_chain = get_qa_chain_for_community(community)

if qa_chain:
    st.success(f"Knowledge base for {community} is ready.")
else:
    st.error(f"Could not initialize the system for {community}.")
    st.stop()

# --- 5. Conversational Chat Interface ---
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}
if community not in st.session_state.chat_histories:
    st.session_state.chat_histories[community] = []

for message in st.session_state.chat_histories[community]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.chat_histories[community].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Finding the answer..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                answer = result.get("result", "No answer could be generated.")
                st.markdown(answer)
                st.session_state.chat_histories[community].append({"role": "assistant", "content": answer})

                if result.get("source_documents"):
                    st.subheader("Sources Used:")
                    for doc in result["source_documents"]:
                        source_name = doc.metadata.get('source', 'N/A').split(os.sep)[-1]
                        page_number = doc.metadata.get('page', 'N/A')
                        with st.expander(f"Source: {source_name}, Page: {page_number}"):
                            st.write(doc.page_content)
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                print(traceback.format_exc())
                st.session_state.chat_histories[community].append({"role": "assistant", "content": error_message})