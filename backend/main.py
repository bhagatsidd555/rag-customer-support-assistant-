from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import upload, chat

app = FastAPI(
    title="AI Customer Support Assistant",
    description="RAG-based support assistant for manufacturing manuals",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])

@app.get("/")
def root():
    return {"message": "AI Customer Support Assistant is running!"}

@app.get("/api/health")
def health():
    return {"status": "ok"}
print("all well")