# Retrieval-Augmented Generation (RAG) Project

This project implements a question-answering system using a Retrieval-Augmented Generation (RAG) architecture. You can load your own documents (in this case, research papers in PDF format) and ask questions about their content.

## How It Works

The project uses a powerful combination of technologies:

* **LangChain**: A framework for developing applications powered by language models.
* **Pinecone**: A vector database for storing and retrieving vector embeddings of your documents.
* **Hugging Face**: Provides pre-trained models for generating text embeddings and for the language model itself.
* **Streamlit**: To create a simple and interactive web interface for asking questions.

The process is as follows:
1.  **Load Documents**: The PDF documents in the `data` directory are loaded.
2.  **Split Text**: The documents are split into smaller, manageable chunks.
3.  **Create Embeddings**: Each chunk of text is converted into a numerical vector (embedding) that captures its semantic meaning.
4.  **Store in Pinecone**: These embeddings are stored in a Pinecone vector database.
5.  **Ask a Question**: When you ask a question, it is also converted into an embedding.
6.  **Retrieve Relevant Information**: Pinecone is used to find the text chunks with embeddings most similar to your question's embedding.
7.  **Generate Answer**: The retrieved text chunks and your original question are passed to a language model, which generates a final, coherent answer.

## Setup Instructions

### Step 1: Get Your Pinecone API Key

1.  Go to the [Pinecone website](https://www.pinecone.io/) and sign up for a free account.
2.  After signing up, you will be taken to your dashboard. On the left-hand menu, click on **API Keys**.
3.  You will find your **API Key** and **Environment**. Copy these values.

### Step 2: Set Up Your Environment Variables

1.  In the root of your `rag_project` directory, create a file named `.env`.
2.  Open the `.env` file and add your Pinecone API key and environment like this:

    ```
    PINECONE_API_KEY="YOUR_API_KEY"
    PINECONE_API_ENV="YOUR_ENVIRONMENT"
    ```
    Replace `"YOUR_API_KEY"` and `"YOUR_ENVIRONMENT"` with the values you copied from your Pinecone dashboard.

### Step 3: Install the Required Packages

Make sure you have Python 3.8 or later installed. Then, open your terminal and run the following command to install all the necessary libraries:

```bash
pip install -r requirements.txt
```

## How to Run the Application

1.  Make sure you have placed all your PDF files in the `data` directory.
2.  Open your terminal and navigate to the root of your `rag_project` directory.
3.  Run the following command:

    ```bash
    streamlit run app.py
    ```

4.  This will start the Streamlit web server and open a new tab in your browser. You can now start asking questions about your documents!

Enjoy exploring your documents! 🚀