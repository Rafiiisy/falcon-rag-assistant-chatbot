# ============================================
# 📌 INFERENCE MODULE FOR RAG CHATBOT
# ============================================

import os
import re
import json
import sys
import numpy as np
import asyncio
import aiohttp
import faiss
from rank_bm25 import BM25Okapi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    API_MAX_TOKENS, API_TEMPERATURE, API_TOP_P, API_STREAM, 
    BM25_K1, BM25_B, PROMPT_TEMPLATE, PLACEHOLDER_REPLACEMENTS, 
    REMOVE_PLACEHOLDERS, API_URL, HEADERS, FAISS_NEIGHBORS_K,
    BM25_CONFIDENCE_THRESHOLD, FAISS_CONFIDENCE_THRESHOLD
)
# ============================================
# 📌 SECTION 1: LOAD FAQ DATASET & EMBEDDINGS
# ============================================

base_dir = os.path.dirname(os.path.abspath(__file__))
faq_path = os.path.join(base_dir, "..", "data", "cleaned_faqs.json")

with open(faq_path, "r") as file:
    faq_data = json.load(file)

tokenized_corpus = [faq["question"].split() for faq in faq_data]
bm25 = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)

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
    distances, indices = index.search(query_embedding, k=FAISS_NEIGHBORS_K)

    bm25_confidence = bm25_scores[bm25_top_idx]
    faiss_confidence = 1 / (1 + distances[0][0])

    # Adjust the thresholds so you can test out-of-scope more easily
    if bm25_confidence > BM25_CONFIDENCE_THRESHOLD or faiss_confidence > FAISS_CONFIDENCE_THRESHOLD:
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
        return f"Hello! How can I assist you today?"

    # Out-of-scope fallback
    return (
        f"Sorry, but that question is outside my current scope. "
        f"I can only help with order, refund, or shipping-related queries at this time. "
        f"Please contact {COMPANY_NAME} support at {SUPPORT_CONTACT} for further assistance."
    )

# ============================================
# 📌 SECTION 3: SET UP CHATPROMPT TEMPLATE
# ============================================

chat_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ============================================
# 📌 SECTION 4: ASYNC API CALLS (NON-STREAMING)
# ============================================

async def chat_with_falcon_api(user_query: str) -> str:
    """Send a structured prompt to Falcon-7B API and return only the relevant answer with cleaned placeholders."""

    retrieved_answer = hybrid_search(user_query)

    # ✅ Directly return greetings without passing to LLM
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if retrieved_answer in greetings or retrieved_answer.startswith("Hello! How can I assist you today?"):
        return retrieved_answer  # 🚀 Return immediately if it's a greeting
    
    # ✅ Continue with LLM processing for non-greeting queries
    final_query = chat_prompt.format(
        user_query=user_query,
        retrieved_answer=retrieved_answer
    )

    payload = {
        "inputs": final_query,
        "parameters": {
            "max_new_tokens": API_MAX_TOKENS,
            "temperature": API_TEMPERATURE,
            "top_p": API_TOP_P,
            "stream": API_STREAM
        }
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(API_URL, headers=HEADERS, json=payload) as response:
                response_text = await response.text()  # Capture full response for debugging

                if response.status == 200:
                    try:
                        data = await response.json()
                        if isinstance(data, list) and data and "generated_text" in data[0]:
                            output_text = data[0]["generated_text"]
                        elif isinstance(data, dict) and "generated_text" in data:
                            output_text = data["generated_text"]
                        else:
                            raise ValueError("❌ Unexpected API Response Format.")

                        # ✅ Extract only the FAQ answer using regex
                        match = re.search(r"We retrieved the following FAQ content.*?Answer:\s*(.*?)\s*\(Category:", output_text, re.DOTALL)
                        extracted_answer = match.group(1).strip() if match else output_text

                        # ✅ Replace known placeholders dynamically from config.py
                        for placeholder, replacement in PLACEHOLDER_REPLACEMENTS.items():
                            extracted_answer = extracted_answer.replace(placeholder, replacement)

                        # ✅ Remove sensitive placeholders from response
                        for placeholder in REMOVE_PLACEHOLDERS:
                            extracted_answer = re.sub(rf"\b{placeholder}\b", "", extracted_answer).strip()

                        return extracted_answer  # ✅ Return final clean response

                    except Exception as e:
                        print("❌ JSON Parsing Error:", str(e))
                        print("❌ Full API Response:", response_text)
                        return "I'm sorry, but I'm unable to process your request at the moment. Please try again later."
                
                elif response.status == 500:
                    print("❌ API Error: 500 (Internal Server Error) - Likely due to out-of-scope input.")
                    return "I'm sorry, but that question is outside my scope. I can help with order, refund, or shipping-related queries."

                else:
                    print(f"❌ API Error: {response.status} - {response_text}")
                    return "I'm sorry, but I couldn't process your request at the moment. Please try again later."

        except Exception as e:
            print("❌ Unexpected Error:", str(e))
            import traceback
            traceback.print_exc()
            return "I'm sorry, but I'm currently experiencing technical difficulties. Please try again later."