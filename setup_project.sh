#!/bin/bash

# Create required folders
mkdir -p data models/llama2_model retriever chatbot api frontend docker

# Create files and add initial content
echo '{}' > data/faqs.json
touch data/policies.pdf

echo "import pickle" > models/embeddings.pkl

echo "import json\n# Preprocessing script for FAQs & policies" > retriever/preprocess.py
echo "from sentence_transformers import SentenceTransformer\n# Generates vector embeddings" > retriever/embeddings.py
echo "import chromadb\n# Setup and store embeddings in ChromaDB" > retriever/db_setup.py
echo "import chromadb\n# Retrieves most relevant documents" > retriever/retriever.py

echo "import ollama\n# Calls LLaMA for response generation" > chatbot/inference.py
echo "import retriever\n# Combines retrieval with LLM inference" > chatbot/chat_pipeline.py

echo "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/')\ndef read_root():\n    return {'message': 'Chatbot API is running'}" > api/main.py

echo "import streamlit as st\nst.title('ECommerce Support Chatbot')" > frontend/app.py

echo "FROM python:3.10\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD ['python', 'api/main.py']" > docker/Dockerfile
echo "version: '3.8'\nservices:\n  chatbot:\n    build: .\n    ports:\n      - '8000:8000'" > docker/docker-compose.yml

echo "openai\nfastapi\nchromadb\nsentence-transformers\nstreamlit" > requirements.txt
echo "# Intelligent RAG Chatbot using LLaMA & LangChain" > README.md
echo "__pycache__/\n.env\nmodels/llama2_model/" > .gitignore

# Make script executable
chmod +x setup_project.sh

echo "✅ Project setup completed! You can now start coding."
