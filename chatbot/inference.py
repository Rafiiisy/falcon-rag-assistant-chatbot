# ============================================
# 📌 INFERENCE MODULE FOR RAG CHATBOT
# ============================================

import os
import re
import json
import numpy as np
import asyncio
import aiohttp
import faiss
from rank_bm25 import BM25Okapi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# ✅ Hugging Face API for Falcon-7B
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Get API key from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise ValueError("❌ API Key is missing! Set HF_API_KEY in your environment.")

# ✅ Use API key securely
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
# ============================================
# 📌 SECTION 1: LOAD FAQ DATASET & EMBEDDINGS
# ============================================

base_dir = os.path.dirname(os.path.abspath(__file__))
faq_path = os.path.join(base_dir, "..", "data", "limited_faqs.json")

with open(faq_path, "r") as file:
    faq_data = json.load(file)

tokenized_corpus = [faq["question"].split() for faq in faq_data]
bm25 = BM25Okapi(tokenized_corpus)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = np.array([embedding_model.embed_query(faq["answer"]) for faq in faq_data])

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ============================================
# 📌 SECTION 2: HYBRID SEARCH (BM25 + FAISS)
# ============================================

def hybrid_search(query: str) -> str:
    """Combine BM25 lexical search with FAISS semantic search for better accuracy."""
    tokenized_query = query.split()
    
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_idx = np.argmax(bm25_scores)

    query_embedding = np.array([embedding_model.embed_query(query)])
    distances, indices = index.search(query_embedding, k=1)

    bm25_confidence = bm25_scores[bm25_top_idx]
    faiss_confidence = 1 / (1 + distances[0][0])

    # Adjust the thresholds so you can test out-of-scope more easily
    if bm25_confidence > 4.0 or faiss_confidence > 0.6:
        if bm25_confidence > faiss_confidence:
            matched_faq = faq_data[bm25_top_idx]
        else:
            matched_faq = faq_data[indices[0][0]]
        return (
            f"Answer: {matched_faq['answer']}\n\n"
            f"(Category: {matched_faq['category']} | Intent: {matched_faq['intent']})"
        )

    # Simple greetings
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if query.lower().strip() in greetings:
        return "Hello! How can I assist you today?"

    # Out-of-scope fallback
    return (
        "Sorry, but that question is outside my current scope. "
        "I can only help with order, refund, or shipping-related queries at this time. "
        "Please contact support for further assistance."
    )

# ============================================
# 📌 SECTION 3: SET UP CHATPROMPT TEMPLATE
# ============================================

# If the retrieved text is the fallback, the LLM is basically told to restate it.
chat_prompt = ChatPromptTemplate.from_template("""
You are a customer support AI. 
Here is the user's question:

User: "{user_query}"

We retrieved the following FAQ content (or fallback message) which might help:
{retrieved_answer}

Instructions:
- If the text above states it's out of scope, restate that it is out of scope. 
- Otherwise, incorporate the FAQ info in your own words (avoid copying verbatim).
- Maintain a polite, professional tone.

Now provide a final answer to the user:
""")

# ============================================
# 📌 SECTION 4: ASYNC API CALLS (NON-STREAMING)
# ============================================

async def chat_with_falcon_api(user_query: str) -> str:
    """Send a structured prompt to Falcon-7B API and print the cleaned response."""
    retrieved_answer = hybrid_search(user_query)
    final_query = chat_prompt.format(
        user_query=user_query,
        retrieved_answer=retrieved_answer
    )

    payload = {
        "inputs": final_query,
        "parameters": {"max_new_tokens": 256, "temperature": 0.7},
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=HEADERS, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    output_text = data[0]["generated_text"]
                elif isinstance(data, dict) and "generated_text" in data:
                    output_text = data["generated_text"]
                else:
                    output_text = "❌ Unexpected API Response Format."

                # Example post-processing for leftover instructions
                output_text = re.sub(r"^Human:\s*", "", output_text, flags=re.IGNORECASE).strip()
                output_text = re.sub(r"You are a customer support AI\..*?Now provide a final answer to the user:\s*", "", output_text, flags=re.DOTALL).strip()

                return output_text
            else:
                return f"❌ API Error: {response.status} - {await response.text()}"
