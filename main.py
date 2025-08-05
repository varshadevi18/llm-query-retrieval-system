import os
import io
import requests
import pdfplumber
import faiss
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException, Header
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

# Initialize app and model
app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

TEAM_TOKEN = "87436cd7e9ec09c6ae1c66eb55aa5da937d1ec6c22a032731eb773c9a9727777"

# ---------------------- PDF Extraction ----------------------
def extract_text_from_pdf_url(url: str) -> List[str]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_bytes = io.BytesIO(response.content)

        docs = []

        def process_page(page):
            text = page.extract_text()
            if text:
                return [para.strip() for para in text.split('\n') if para.strip()]
            return []

        with pdfplumber.open(file_bytes) as pdf:
            with ThreadPoolExecutor() as executor:
                results = executor.map(process_page, pdf.pages)
            for page_result in results:
                docs.extend(page_result)

        return docs
    except Exception as e:
        print(f"PDF Error: {e}")
        return []

# ---------------------- FAISS Index ----------------------
def build_faiss_index(docs: List[str]):
    embeddings = model.encode(docs, batch_size=16, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

# ---------------------- Match Closest Clause ----------------------
def find_best_match(index, docs, question: str, top_k=3) -> str:
    question_embedding = model.encode([question], show_progress_bar=False)
    distances, indices = index.search(np.array(question_embedding), k=top_k)
    top_matches = [docs[i] for i in indices[0] if i < len(docs)]
    best = sorted(top_matches, key=lambda x: -len(set(x.lower().split()) & set(question.lower().split())))[0]
    return best

# ---------------------- Intent Parser ----------------------
def llm_parser(q: str):
    return {"intent": "lookup_clause", "focus": q}

# ---------------------- Logic Evaluation ----------------------
def logic_decision(text: str) -> str:
    lower = text.lower()
    if "not covered" in lower or "excluded" in lower:
        return "rejected"
    elif "covered" in lower or "indemnify" in lower:
        return "approved"
    else:
        return "uncertain"

# ---------------------- API Models ----------------------
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    success: bool
    answers: List[str]

# ---------------------- Main Endpoint ----------------------
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
def hackrx_run(request: QueryRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {TEAM_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")

    docs = extract_text_from_pdf_url(request.documents)
    if not docs:
        raise HTTPException(status_code=400, detail="Document extraction failed")

    index = build_faiss_index(docs)
    answers = []

    for q in request.questions:
        parsed = llm_parser(q)
        best_clause = find_best_match(index, docs, parsed["focus"])
        short_clause = best_clause.strip().replace("\n", " ")
        if len(short_clause) > 400:
            short_clause = short_clause[:400] + "..."
        answers.append(short_clause)

    return {"success": True, "answers": answers}

# ---------------------- Root Endpoint ----------------------
@app.get("/")
def root():
    return {"message": "LLM Query Retrieval API is up and running 🚀"}

# ---------------------- Swagger Auth ----------------------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="LLM Query Retrieval API",
        version="1.0.0",
        description="HackRx 6.0 - PDF Clause Extraction & Query Answering",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization"
        }
    }
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# ---------------------- Run Server ----------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
