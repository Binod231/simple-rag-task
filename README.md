# Financial Document Explorer

This project is an interactive, multi-turn conversational RAG system that allows you to "chat" with financial documents (specifically 10-K reports). You can ask complex questions in plain English and receive answers sourced directly from the documents with citations.

## How It Works

1.  **Community Selection**: Users select a company (a "community") from a dropdown, scoping the search to that company's documents.
2.  **Document Loading**: 10-K reports (PDFs) from the selected community's `data/` sub-directory are loaded.
3.  **Text Chunking**: The text is split into smaller, semantically meaningful chunks.
4.  **Vector Embeddings**: Each chunk is converted into a numerical vector embedding.
5.  **Vector Storage**: These embeddings are stored and indexed in a **ChromaDB** local vector database.
6.  **Conversational Retrieval**: When you ask a question, the system retrieves the most relevant text chunks from the selected community's documents.
7.  **Answer Generation**: The retrieved chunks and your question are passed as context to a local LLM (run via **Ollama**) to generate a comprehensive answer with source citations.

## Tech Stack

* **Framework**: [Streamlit](https://streamlit.io/)
* **Orchestration**: [LangChain](https://www.langchain.com/)
* **Vector Database**: [ChromaDB](https://www.trychroma.com/)
* **LLM Hosting**: [Ollama](https://ollama.com/) (with Llama 3)
* **Embeddings**: [Hugging Face](https://huggingface.co/) Sentence Transformers

## Setup and Installation

### Prerequisites

* Python 3.9+
* [Ollama](https://ollama.com/) installed and running.
* The required LLM pulled via Ollama. Open your terminal and run:
    ```bash
    ollama pull llama3
    ```

### Step 1: Install Dependencies

1.  Create and activate a Python virtual environment.
2.  Install all required packages from the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

1.  Organize your PDF files into sub-directories within the `data/` directory. Each sub-directory will be a "community."
2.  Make sure the Ollama application is running in the background.
3.  Open your terminal, navigate to the project root, and run:

    ```bash
    streamlit run app.py
    ```

4.  The application will open in a new browser tab, ready for you to ask questions!

## Project Structure
```bash
rag-project/
├── app.py                # The main Streamlit application script
├── config.py             # Configuration for the LLM model
├── requirements.txt      # Python package dependencies
├── README.md             # This file
└── data/
    ├── Apple/
    │   └── apple-10k.pdf
    ├── Nvidia/
    │   └── nvidia-10k.pdf
    └── Tesla/
        └── tesla-10k.pdf
        ```
```
## Troubleshooting & Performance
Why is the "Finding the answer..." step slow?
The response time depends heavily on your computer's hardware. The slowdown occurs because the application sends a large amount of context (retrieved text from the PDF) to the local LLM. Running this on a CPU is very demanding.

To improve speed:

Use a GPU: If you have a dedicated NVIDIA GPU, Ollama will automatically use it, making responses significantly faster.

Use a Smaller Model: In config.py, switch to a smaller model like "gemma:2b" or "tinyllama". This is the most effective way to improve speed on CPU.```       
