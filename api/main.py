# ============================================
# 📌 FASTAPI BACKEND FOR CHATBOT
# ============================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chatbot.inference import chat_with_falcon_api

# ✅ Initialize FastAPI
app = FastAPI(
    title="Falcon-7B Chatbot API",
    description="API for hybrid search-based chatbot using Falcon-7B, FAISS, and BM25",
    version="1.0"
)

# ✅ Define request body model
class ChatRequest(BaseModel):
    query: str

# ============================================
# 📌 SECTION 1: CHATBOT ENDPOINT
# ============================================

@app.post("/chat/")
async def chat(request: ChatRequest):
    """Endpoint to receive user query and return chatbot response."""
    try:
        response = await chat_with_falcon_api(request.query)
        return {"user_query": request.query, "chatbot_response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# 📌 SECTION 2: ROOT ENDPOINT
# ============================================

@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"message": "Chatbot API is running. Send a POST request to /chat/ with {'query': 'your message'}"}

# ============================================
# 📌 SECTION 3: RUN FASTAPI (For local testing)
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
