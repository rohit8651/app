from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faiss_search import ask_question, search_faiss, query_company_ai
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Enable CORS (Fixes XMLHttpRequest error in Flutter Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root Endpoint
@app.get("/")
def root():
    return {"message": "AI Document Search API is running!"}

# GET request for asking a question
@app.get("/ask")
def get_ask(query: str):
    try:
        return {"response": ask_question(query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Request model for POST request
class QueryRequest(BaseModel):
    query: str

# POST request for asking a question
@app.post("/ask")
def post_ask(request: QueryRequest):
    try:
        status, retrieved_chunks = search_faiss(request.query)

        if status == "Out of context":
            return {"response": "I cannot answer that based on the provided information."}

        ai_response = query_company_ai(request.query, retrieved_chunks)
        return {"response": ai_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
