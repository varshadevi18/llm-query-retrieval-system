import os
import requests
import pdfplumber
import faiss
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException, Header
from fastapi.openapi.models import APIKey, APIKeyIn, SecuritySchemeType
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Initialize app and model
app = FastAPI()
model = SentenceTransformer("'paraphrase-MiniLM-L3-v2")

# Auth token from HackRx portal
TEAM_TOKEN = "87436cd7e9ec09c6ae1c66eb55aa5da937d1ec6c22a032731eb773c9a9727777"

# ------------- Step 1: Extract Text from PDF URL -------------
def extract_text_from_pdf_url(url: str) -> List[str]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        docs = []
        with pdfplumber.open("temp.pdf") as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    docs.append(text.strip())
        os.remove("temp.pdf")
        return docs
    except Exception as e:
        print(f"❌ PDF Error: {e}")
        return []

# ------------- Step 2: Build Embedding Search Index -------------
def build_faiss_index(docs: List[str]):
    embeddings = model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

# ------------- Step 3: Semantic Clause Matching -------------
def find_best_match(index, docs, question: str) -> str:
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), k=1)
    best_idx = indices[0][0]
    return docs[best_idx]

# ------------- Step 4: (Stub) Query Parser -------------
def llm_parser(q: str):
    return {"intent": "lookup_clause", "focus": q}

# ------------- Step 5: Logic Evaluation -------------
def logic_decision(text: str) -> str:
    lower = text.lower()
    if "not covered" in lower or "excluded" in lower:
        return "rejected"
    elif "covered" in lower or "indemnify" in lower:
        return "approved"
    else:
        return "uncertain"

# ------------- Step 6: API Request & Response Models -------------
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# ------------- Final API Endpoint -------------
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
def hackrx_run(request: QueryRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {TEAM_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid token")

    docs = extract_text_from_pdf_url(request.documents)
    if not docs:
        raise HTTPException(status_code=400, detail="Document processing failed")

    index = build_faiss_index(docs)

    answers = []
    for q in request.questions:
        parsed = llm_parser(q)
        best_clause = find_best_match(index, docs, parsed["focus"])
        decision = logic_decision(best_clause)
        answer = best_clause.strip().replace("\n", " ")
        answers.append(answer)

    return {"answers": answers}

@app.get("/")
def root():
    return {"message": "LLM Query Retrieval API is up and running 🚀"}

# ------------- Swagger Authorize Button Configuration -------------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="LLM Query Retrieval API",
        version="1.0.0",
        description="HackRx 6.0 API for clause extraction and analysis",
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
