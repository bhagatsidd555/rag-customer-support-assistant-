import os
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv


import chromadb
from chromadb.config import Settings
from groq import Groq
from sentence_transformers import SentenceTransformer
import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

CHROMA_DIR = Path("chroma_db")
CHROMA_DIR.mkdir(exist_ok=True)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHAT_MODEL   = "llama-3.1-8b-instant"
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 100

# Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Local embedding model (free, no Ollama needed)
logger.info("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Embedding model loaded.")

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(
    name="manuals",
    metadata={"hnsw:space": "cosine"}
)

conversation_memory: Dict[str, List[Dict]] = {}


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_embedding(text: str) -> List[float]:
    return embed_model.encode(text, normalize_embeddings=True).tolist()


def ingest_pdf(file_path: str, filename: str) -> Dict[str, Any]:
    logger.info(f"Ingesting PDF: {filename}")

    raw_text = extract_text_from_pdf(file_path)
    if not raw_text.strip():
        raise ValueError("No text extracted from PDF.")

    chunks = chunk_text(raw_text)
    logger.info(f"Created {len(chunks)} chunks")

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        ids.append(f"{filename}_{i}_{uuid.uuid4().hex[:8]}")
        embeddings.append(emb)
        documents.append(chunk)
        metadatas.append({"source": filename, "chunk_index": i})

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.info(f"Stored {len(chunks)} chunks in vector DB")

    return {"filename": filename, "chunks": len(chunks), "characters": len(raw_text)}


def retrieve_context(query: str, top_k: int = 5) -> List[Dict]:
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    contexts = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        contexts.append({
            "text": doc,
            "source": meta.get("source", "Unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "relevance_score": round(1 - dist, 4)
        })
    return contexts


def build_prompt(query: str, contexts: List[Dict], history: List[Dict]) -> List[Dict]:
    context_text = "\n\n".join(
        [f"[Source: {c['source']}, Chunk {c['chunk_index']}]\n{c['text']}" for c in contexts]
    )

    system_msg = {
        "role": "system",
        "content": (
            "You are an expert AI Customer Support Assistant for a manufacturing company. "
            "Answer questions ONLY based on the provided manual context. "
            "Always cite the source document and chunk number when you use information from it. "
            "If the answer is not in the context, say: 'I could not find this in the uploaded manuals.' "
            "Be concise, accurate, and helpful.\n\n"
            f"CONTEXT FROM MANUALS:\n{context_text}"
        )
    }

    messages = [system_msg] + history[-6:] + [{"role": "user", "content": query}]
    return messages


def chat_with_rag(session_id: str, query: str) -> Dict[str, Any]:
    logger.info(f"Query [{session_id}]: {query}")

    contexts = retrieve_context(query)
    if not contexts:
        return {
            "answer": "No manuals have been uploaded yet. Please upload a PDF manual first.",
            "sources": [],
            "session_id": session_id
        }

    history = conversation_memory.get(session_id, [])
    messages = build_prompt(query, contexts, history)

    # Groq chat call
    response = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )
    answer = response.choices[0].message.content

    conversation_memory.setdefault(session_id, [])
    conversation_memory[session_id].append({"role": "user", "content": query})
    conversation_memory[session_id].append({"role": "assistant", "content": answer})

    if len(conversation_memory[session_id]) > 20:
        conversation_memory[session_id] = conversation_memory[session_id][-20:]

    sources = list({c["source"] for c in contexts})
    return {
        "answer": answer,
        "sources": sources,
        "contexts": contexts[:3],
        "session_id": session_id
    }


def list_documents() -> List[str]:
    results = collection.get(include=["metadatas"])
    sources = list({m.get("source", "") for m in results["metadatas"] if m.get("source")})
    return sorted(sources)


def delete_document(filename: str) -> bool:
    results = collection.get(include=["metadatas"])
    ids_to_delete = [
        id_ for id_, meta in zip(results["ids"], results["metadatas"])
        if meta.get("source") == filename
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        return True
    return False