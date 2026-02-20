# Building RAG Agents with LLMs (NVIDIA DLI)

This repository contains the coursework and exercises for the NVIDIA Deep Learning Institute course on building Retrieval-Augmented Generation (RAG) agents. The codebase demonstrates how to orchestrate Large Language Models (LLMs) using **LangChain**, **NVIDIA AI Foundation Endpoints**, and **Vector Stores** to create intelligent, context-aware applications.

## Course Overview

The curriculum is divided into 8 notebooks that progress from basic environment setup to deploying a fully evaluated RAG agent.

### 1. Environment & Microservices
**Notebook:** `01_environment.ipynb`
*   **Focus:** Understanding the containerized environment and microservice orchestration.
*   **Key Concepts:** Interacting with Docker containers, understanding the Jupyter Lab microservice, and connecting to the frontend and LLM client services.
*   **Tools:** Docker, Requests, Curl.

### 2. LLM Services & Foundation Models
**Notebook:** `02_llm_services.ipynb`
*   **Focus:** Interfacing with NVIDIA AI Foundation Models (NIM) for inference.
*   **Key Concepts:** Switching between manual Python requests and the `ChatNVIDIA` LangChain connector. Accessing models like Llama 3 and Mistral via API.
*   **Tools:** `langchain_nvidia_ai_endpoints`, `requests`.

### 3. LangChain Expression Language (LCEL)
**Notebook:** `03_langchain_lcel.ipynb`
*   **Focus:** Mastering the modern LangChain syntax for orchestration.
*   **Key Concepts:** Building Chains and Runnables, using the pipe (`|`) operator, and creating simple chat interfaces with Gradio.
*   **Exercises:** Creating a "Rhyme Re-themer" chatbot.

### 4. Running State Chains & Dialog Management
**Notebook:** `04_state_chains.ipynb`
*   **Focus:** Managing conversation history and internal state.
*   **Key Concepts:** "Running State Chains" to maintain context, using Pydantic for knowledge base slot-filling, and routing logic.
*   **Exercise:** Building an Airline Customer Service Bot that gates sensitive database information.

### 5. Reasoning with Large Documents
**Notebook:** `05_large_docs.ipynb`
*   **Focus:** Handling documents that exceed the LLM context window.
*   **Key Concepts:** Document loading (Arxiv), chunking strategies, and "Document Refinement" (progressive summarization).
*   **Tools:** `ArxivLoader`, `RecursiveCharacterTextSplitter`.

### 6. Embeddings & Semantic Reasoning
**Notebook:** `06_embeddings.ipynb`
*   **Focus:** Understanding how text is converted into vectors for comparison.
*   **Key Concepts:** Using `NVIDIAEmbeddings` for "Query" vs. "Passage" embedding, calculating Cosine Similarity, and visualizing semantic relationships.

### 7. RAG with Vector Stores
**Notebook:** `07_rag_vectors.ipynb`
*   **Focus:** Building the complete RAG pipeline.
*   **Key Concepts:** Ingesting documents into FAISS vector stores, creating Retrievers, and implementing "Always-on RAG" chains.
*   **Exercise:** Implementing a RAG chain capable of discussing specific Arxiv papers.

### 8. RAG Evaluation & Assessment
**Notebook:** `08_evaluation.ipynb`
*   **Focus:** Testing and deploying the pipeline.
*   **Key Concepts:** "LLM-as-a-Judge" methodology (using synthetic data to grade performance) and deploying the chain as a microservice using **LangServe** and **FastAPI**.
*   **Final Assessment:** The agent must pass an automated assessment by exposing endpoints (`/basic_chat`, `/retriever`, `/generator`) on port 9012.

Based on **Notebook 8: RAG Evaluation**, the final assessment focuses on two major tasks: **evaluating your RAG pipeline** using an "LLM-as-a-Judge" approach and **deploying your pipeline** as a microservice to be graded by an automated system.

### 1. The Evaluation Strategy: "LLM-as-a-Judge"
Before submitting your work for grading, the notebook guides you through a self-evaluation process to ensure your RAG system is actually effective. Instead of manually reading hundreds of answers, you implemented an automated workflow where a powerful LLM acts as a "Judge" to score your system.

This workflow involved four distinct steps:
*   **Sampling:** I randomly selected two document chunks from your `docstore_index` (the vector index you created in Notebook 7).
*   **Synthetic Data Generation:** You used an LLM to read those chunks and generate a "Ground Truth" Question-Answer pair. This effectively creates a test case where you *know* the correct answer because it was derived directly from the source text.
*   **RAG Generation:** I fed *only* the synthetic question into your RAG chain to see what answer it produced on its own.
*   **Judgement:** I passed three pieces of information to the "Judge" LLM: the Question, the Ground Truth Answer, and your RAG Agent's Answer. The Judge was instructed to score your agent's answer based on whether it matched the accuracy of the ground truth.

### 2. Recreating the RAG Pipeline
To perform this evaluation, I had to reconstruct the RAG pipeline I learned in previous lessons. This involved:
*   **Loading the Index:** I used `FAISS.load_local` to retrieve the vector store containing your Arxiv papers.
*   **Defining the Retriever:** I created a retrieval chain that searches for the top 4 relevant documents based on a user query.
*   **Defining the Generator:** I set up a `ChatNVIDIA` model (Llama 3.1) to take those retrieved documents and answer the user's question conversationally.

### 3. The Final Exam: Deploying for Assessment
The actual "grade" for the course comes from an external autograder that queries your system. To facilitate this, you had to convert your RAG chain into a running web server.

*   **Server Setup:**  Utilized `FastAPI` and `LangServe` to create an API server.
*   **Port Configuration:** The server was explicitly bound to **port 9012**, which is where the course's grading system looks for your agent.
*   **Required Endpoints:** Expose three specific routes for the grader to test:
    *   `/basic_chat`: A standard LLM chain without retrieval, used as a baseline.
    *   `/retriever`: The retrieval chain, allowing the grader to verify that you are actually pulling relevant documents from the index.
    *   `/generator` (or `/rag`): The full RAG chain, allowing the grader to test end-to-end question answering.

### 4. Verification & Submission
The final step involved killing any old server processes (using `pkill`) and starting your new `server_app.py` script. Once the server was running on port 9012, you clicked the **"Assess Task"** button in the course environment. The autograder then sent test queries to your API endpoints and verified that your system could correctly retrieve information and generate accurate answers from the documents you indexed.

## Tech Stack

*   **Orchestration:** [LangChain](https://python.langchain.com/) (Python)
*   **Models:** NVIDIA AI Foundation Endpoints (Llama 3, Mistral, etc.)
*   **Vector Store:** FAISS (CPU)
*   **Deployment:** FastAPI / LangServe
*   **UI:** Gradio
*   **Containerization:** Docker

## Usage

1.  **Dependencies:** Ensure you have the required packages installed (included in course environment):
    ```python
    %pip install langchain langchain-nvidia-ai-endpoints gradio faiss-cpu
    ```
2.  **API Keys:** Set your `NVIDIA_API_KEY` if running outside the managed environment.
3.  **Deployment:** To run the final agent for assessment, execute the server app:
    ```bash
    python frontend/server_app.py
    ```
    This launches the API on port 9012.
