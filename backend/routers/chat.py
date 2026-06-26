import uuid
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional  
from services.rag_service import chat_with_rag, retrieve_context, conversation_memory
import ollama

router = APIRouter()


# ── MODELS UPDATED HERE ──
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  


class StreamRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  


@router.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = chat_with_rag(session_id, request.query)
    return result


@router.post("/chat/stream")
async def chat_stream(request: StreamRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    contexts = retrieve_context(request.query)
    if not contexts:
        async def no_docs():
            yield f"data: {json.dumps({'token': 'No manuals uploaded yet.', 'done': False})}\n\n"
            yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'sources': []})}\n\n"
        return StreamingResponse(no_docs(), media_type="text/event-stream")

    context_text = "\n\n".join(
        [f"[Source: {c['source']}, Chunk {c['chunk_index']}]\n{c['text']}" for c in contexts]
    )
    history = conversation_memory.get(session_id, [])
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI Customer Support Assistant for a manufacturing company. "
                "Answer ONLY based on provided context. Cite sources. "
                "If answer not found, say so.\n\nCONTEXT:\n" + context_text
            )
        }
    ] + history[-6:] + [{"role": "user", "content": request.query}]

    sources = list({c["source"] for c in contexts})
    full_answer = []

    async def token_stream():
        stream = ollama.chat(
            model="llama3.2",
            messages=messages,
            stream=True
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            full_answer.append(token)
            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

        answer = "".join(full_answer)
        conversation_memory.setdefault(session_id, [])
        conversation_memory[session_id].append({"role": "user", "content": request.query})
        conversation_memory[session_id].append({"role": "assistant", "content": answer})

        yield f"data: {json.dumps({'done': True, 'session_id': session_id, 'sources': sources})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@router.delete("/chat/{session_id}/history")
def clear_history(session_id: str):
    if session_id in conversation_memory:
        del conversation_memory[session_id]
        return {"message": "Conversation history cleared."}
    raise HTTPException(status_code=404, detail="Session not found.")


@router.get("/chat/{session_id}/history")
def get_history(session_id: str):
    history = conversation_memory.get(session_id, [])
    return {"session_id": session_id, "history": history, "turns": len(history) // 2}