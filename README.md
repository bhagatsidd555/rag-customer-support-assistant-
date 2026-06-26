# AI Customer Support Assistant

## About

This project is a simple AI Customer Support Assistant built using the RAG (Retrieval-Augmented Generation) approach.

The application allows users to upload PDF manuals and ask questions related to the uploaded documents. Instead of answering from general knowledge, the application searches the uploaded manual, retrieves the most relevant information and generates an answer using a local LLM through Ollama.

## Technologies Used

* Python
* FastAPI
* Ollama
* Llama 3.2
* nomic-embed-text
* ChromaDB
* pdfplumber
* HTML
* CSS
* JavaScript

---

## Project Structure

text
rag-customer-support-assistant/

backend/
    main.py
    requirements.txt
    routers/
    services/

frontend/
    index.html

README.md


---

## Setup

### Clone the repository

bash
git clone https://github.com/bhagatsidd555/rag-customer-support-assistant.git


### Move to backend

bash
cd rag-customer-support-assistant/backend


### Create virtual environment

Windows

bash
python -m venv venv
venv\Scripts\activate


macOS / Linux

bash
python3 -m venv venv
source venv/bin/activate


### Install dependencies

bash
pip install -r requirements.txt


### Install Ollama models

bash
ollama pull llama3.2
ollama pull nomic-embed-text


### Run the application

bash
python -m uvicorn main:app --reload --port 8000


Open the backend:


http://127.0.0.1:8000


Swagger API:


http://127.0.0.1:8000/docs


For the frontend, open `frontend/index.html` using Live Server or any local web server.


## Features

* Upload PDF manuals
* Extract text from PDF
* Create embeddings
* Store embeddings in ChromaDB
* Semantic search using vector embeddings
* Generate answers using Llama 3.2
* Show source document
* Session-based conversation history
* Streaming chat response



## API Endpoints

* POST `/api/upload`
* GET `/api/documents`
* DELETE `/api/documents/{filename}`
* POST `/api/chat`
* POST `/api/chat/stream`
* GET `/api/chat/{session_id}/history`
* DELETE `/api/chat/{session_id}/history`
* GET `/api/health`

---

## Workflow


Upload PDF
      ↓
Extract Text
      ↓
Create Chunks
      ↓
Generate Embeddings
      ↓
Store in ChromaDB
      ↓
Retrieve Relevant Chunks
      ↓
Generate Response using Llama 3.2




## Sample Questions

* What is the company name?
* Where is the company located?
* What products are mentioned?
* Which embedding model is used?
* Which vector database is used?
* What is the support email?
* What technologies are mentioned in the document?





