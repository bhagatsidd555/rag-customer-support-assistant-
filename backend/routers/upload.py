import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.rag_service import ingest_pdf, list_documents, delete_document, UPLOAD_DIR

router = APIRouter()


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_pdf(str(save_path), file.filename)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    return {
        "message": f"'{file.filename}' uploaded and processed successfully.",
        "details": result
    }


@router.get("/documents")
def get_documents():
    docs = list_documents()
    return {"documents": docs, "count": len(docs)}


@router.delete("/documents/{filename}")
def remove_document(filename: str):
    deleted = delete_document(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        os.remove(file_path)
    return {"message": f"'{filename}' deleted successfully."}
