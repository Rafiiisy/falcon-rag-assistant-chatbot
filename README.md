# 📌 Intelligent RAG Chatbot with LLAMA Falcon-7B

## Table of Contents
- [Introduction](#introduction)
- [Project Goals & Scope](#project-goals--scope)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Chatbot](#running-the-chatbot)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

## Introduction
Welcome to the **Intelligent RAG Chatbot**! 🚀 This project implements a **Retrieval-Augmented Generation (RAG)** chatbot using BM25 for lexical search, FAISS for semantic retrieval, and Falcon-7B for response generation. It efficiently handles customer support queries by retrieving the most relevant FAQs and generating informative responses.

📌 **Dataset Used:**  
This project is built using a customer support dataset from Kaggle:  
[Bitext Gen AI Chatbot Customer Support Dataset](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset)


## Project Goals & Scope
### Goals
- Develop a chatbot that combines **retrieval-based and generative AI** approaches.
- Provide **accurate, context-aware** responses for customer support.
- Enable **flexible and configurable** chatbot behavior through `config.py`.

### Scope
- Designed for **customer support applications**, utilizing a pre-defined FAQ dataset.
- Supports **FAQ-based retrieval** and **open-ended response generation**.
- Focuses on **efficiency, scalability, and adaptability** through modular design.

## Features
- **Hybrid Search:** BM25 (keyword-based) + FAISS (semantic-based) retrieval.
- **Customizable Responses:** Modify `config.py` to adjust chatbot behavior.
- **Streamlit Web UI:** Easy-to-use frontend for interactive chatbot experience.
- **Evaluation Module:** Measure chatbot performance against ground-truth responses.

## Prerequisites
- Python 3.8+
- Install dependencies using `requirements.txt`
- A **Hugging Face API Key** is required to run the chatbot. Save it in a `.env` file:
  ```bash
  echo "HF_API_KEY=your_huggingface_api_key" > .env
  ```
- Keep in mind that large-scale testing might require a **PRO account** on Hugging Face.

## Installation
```bash
git clone https://github.com/yourusername/intelligent-rag-chatbot.git
cd intelligent-rag-chatbot
python3 -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
pip install -r requirements.txt
```

## Running the Chatbot
To start the backend API:
```bash
uvicorn api.main:app --reload
```

To launch the **Streamlit GUI**:
```bash
python -m streamlit run frontend/app.py
```

## Configuration
You can explore the **capabilities of the chatbot** by tweaking parameters in `config.py`. Adjust retrieval thresholds, API parameters, and placeholder replacements to optimize chatbot responses.

## Evaluation
To evaluate chatbot performance:
```bash
python3 evaluate_chatbot.py
```
This runs an automated test against sampled FAQ queries and calculates similarity scores between chatbot responses and expected answers.

⚠️ **Note:** Running evaluations on a large dataset may require a **PRO account** on Hugging Face due to inference limits.

## Contributing
Feel free to open issues, suggest improvements, or contribute to this project. PRs are welcome! 🚀
