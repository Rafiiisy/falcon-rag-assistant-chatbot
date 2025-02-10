# 🚀 **Falcon RAG Assistant Chatbot**
## 📝 **Overview**

Falcon RAG Assistant Chatbot is an intelligent customer support chatbot built using a Retrieval-Augmented Generation (RAG) pipeline powered by BM25, FAISS, and Falcon-7B. The chatbot provides real-time query responses by leveraging both lexical and semantic search methods.

### 🎯 **Goals & Scope**  
#### ✅ **Goals**
- Implement **hybrid retrieval (BM25 + FAISS)** for **better accuracy** in matching queries.  
- Use **Falcon-7B** as an **LLM-based response generator**.  
- Improve chatbot speed with **parallel API calls**.  
- Provide a **real-time chat interface** using **Streamlit**.  

#### ❌ **Out of Scope**
- The chatbot currently supports **customer service-related queries**.  
- **Not fine-tuned** on a custom dataset.  
- No **multilingual support** (currently only in **English**).  


---

## 📌**Features**

- 🔍 Hybrid Search (BM25 + FAISS) for accurate response retrieval.

- ⚡ FastAPI Backend for seamless API integration.

- 🖥 Streamlit UI for an interactive chatbot interface.

- 🌍 Pretrained Falcon-7B Model for high-quality language generation.

- 📂 Efficient Vector Search with FAISS for semantic understanding.

- 📊 Scalable and Deployable on cloud or local environments.

### ⚙️ **Tech Stack**  
- **LLM Model**: Falcon-7B (via Hugging Face API)  
- **Retrieval**: FAISS (vector search) + BM25 (lexical search)  
- **Backend**: FastAPI (Python)  
- **Frontend**: Streamlit  
- **Data Processing**: Pandas, NumPy  
- **Async API Calls**: Aiohttp  
- **Deployment**: (Local, Cloud options can be added later)  

### 📁 **Project Structure**
```
📦 intelligent-rag-chatbot-llama
├── chatbot/
│   ├── chatbot.py  # Chatbot logic
│   ├── inference.py  # Hybrid search and LLM inference
├── api/
│   ├── main.py  # FastAPI backend
├── frontend/
│   ├── app.py  # Streamlit UI
├── data/
│   ├── limited_faqs.json  # Processed dataset
├── models/
│   ├── falcon/  # Placeholder for model files (if needed)
├── requirements.txt  # Dependencies
├── .gitignore  # Ignore sensitive files
├── README.md  # Project documentation
```
---

### 📸 UI Preview

📌 Screenshot of the chatbot interface in Streamlit.

### 🏗️ Flowchart

📌 Diagram showing the chatbot's decision-making flow.

### 🏛️ System Architecture

📌 Overview of the backend and frontend architecture.

---

### 📊 Dataset Source

The dataset used for training and retrieval is sourced from:🔗 Bitext Gen AI Chatbot Customer Support Dataset

---

## 🚀 Installation & Setup

1️⃣ Clone the Repository
```
git clone https://github.com/Rafiiisy/falcon-rag-assistant-chatbot.git
cd falcon-rag-assistant-chatbot
```
2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
3️⃣ Set Up API Keys (Hugging Face)

Create a **.env** file in the root directory and add your Hugging Face API key:
```
HF_API_KEY=your_huggingface_api_key
```
4️⃣ Run the Backend (FastAPI)
```
python api/main.py
```
5️⃣ Run the Frontend (Streamlit)
```
python -m streamlit run frontend/app.py
```
---

## 🛠️ Future Improvements

✅ Add Streaming Responses for real-time text generation.

✅ Improve Prompt Engineering for better chatbot behavior.

✅ Deploy using Docker or cloud services.

---
## 📜 License

This project is open-source and available under the MIT License.

