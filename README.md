# AI Research Explorer 

This project is an interactive web application that allows you to "chat" with foundational AI research papers. Using a Retrieval-Augmented Generation (RAG) architecture, you can ask complex questions in plain English and receive answers sourced directly from the documents.

## How It Works

The application leverages a modern AI stack to provide a seamless question-answering experience:

1.  **Document Loading**: Research papers (PDFs) from the `data/` directory are loaded and parsed.
2.  **Text Chunking**: The text from each document is split into smaller, semantically meaningful chunks.
3.  **Vector Embeddings**: Each chunk is converted into a numerical vector embedding using a Hugging Face sentence-transformer model.
4.  **Vector Storage**: These embeddings are stored and indexed in a **Pinecone** serverless vector database for fast and efficient retrieval.
5.  **Question & Retrieval**: When you ask a question, it's also converted into an embedding. The system then queries Pinecone to find the most relevant text chunks from the documents.
6.  **Answer Generation**: The retrieved chunks and your original question are passed as context to a local Large Language Model (run via **Ollama**) which generates a comprehensive answer.

## Tech Stack

* **Framework**: [Streamlit](https://streamlit.io/)
* **Orchestration**: [LangChain](https://www.langchain.com/)
* **Vector Database**: [Pinecone](https://www.pinecone.io/)
* **LLM Hosting**: [Ollama](https://ollama.com/) (with Llama 3)
* **Embeddings**: [Hugging Face](https://huggingface.co/) Sentence Transformers

## Setup and Installation

Follow these steps to get the application running on your local machine.

### Prerequisites

* Python 3.9+
* [Ollama](https://ollama.com/) installed and running.
* The required LLM pulled via Ollama. Open your terminal and run:
    ```bash
    ollama pull llama3
    ```

### Step 1: Get Your Pinecone API Key

1.  Create a free account on the [Pinecone website](https://www.pinecone.io/).
2.  In your dashboard, navigate to the **API Keys** section.
3.  Copy your **API Key** and **Environment** values (e.g., `us-east-1`).

### Step 2: Set Up Environment Variables

1.  In the project's root directory, create a file named `.env`.
2.  Add your Pinecone credentials to this file:

    ```
    PINECONE_API_KEY="YOUR_API_KEY"
    PINECONE_API_ENV="YOUR_ENVIRONMENT"
    ```

### Step 3: Install Dependencies

1.  Create and activate a Python virtual environment.
2.  Install all required packages from the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

1.  Ensure your PDF files are placed in the `data/` directory.
2.  Make sure the Ollama application is running in the background.
3.  Open your terminal, navigate to the project root, and run:

    ```bash
    streamlit run app.py
    ```

4.  The application will open in a new browser tab, ready for you to ask questions!

## Project Structure

```
rag-project/
├── .env                  # Stores API keys and environment variables
├── app.py                # The main Streamlit application script
├── config.py             # Configuration for models and Pinecone index
├── requirements.txt      # Python package dependencies
├── README.md             # This file
└── data/
    ├── AttentionIsAllYouNeed.pdf
    ├── LLaMA.pdf
    └── ... (and other research papers)
```