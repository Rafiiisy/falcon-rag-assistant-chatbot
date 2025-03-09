import json
import random
import asyncio
import numpy as np
import torch
from chatbot.inference import chat_with_falcon_api
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

# ✅ Load dataset
def load_faqs(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# ✅ Organize dataset by intent (limit 5 test queries per intent)
def prepare_test_data(faq_data):
    intent_groups = defaultdict(list)
    for faq in faq_data:
        intent_groups[faq["intent"]].append(faq)
    
    test_queries = {}
    for intent, faqs in intent_groups.items():
        test_queries[intent] = random.sample(faqs, min(5, len(faqs)))  # Max 5 queries per intent
    
    return test_queries

# ✅ Compute similarity (Semantic Evaluation)
def evaluate_similarity(generated_response, expected_responses):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    gen_embedding = model.encode(generated_response, convert_to_tensor=True)
    expected_embeddings = model.encode(expected_responses, convert_to_tensor=True)

    # 🔹 Convert tensor to CPU before using numpy
    similarities = util.pytorch_cos_sim(gen_embedding, expected_embeddings)
    max_similarity = np.max(similarities.cpu().numpy())  # Fix: Move tensor to CPU

    return max_similarity

# ✅ Run chatbot evaluation
async def evaluate_chatbot():
    print("🚀 Loading FAQ dataset...")
    faq_data = load_faqs("data/limited_faqs.json")
    test_data = prepare_test_data(faq_data)
    print(f"✅ Dataset loaded with {len(test_data)} intents.")

    results = {}

    for intent, faqs in test_data.items():
        test_questions = [faq["question"] for faq in faqs]
        expected_answers = [faq["answer"] for faq in faqs]

        print(f"\n🔎 Testing intent: {intent} with {len(test_questions)} queries...")

        test_question = random.choice(test_questions)

        try:
            chatbot_response = await chat_with_falcon_api(test_question)

            # ✅ Handle API limit exceeded (Error 402)
            if "API Error: 402" in chatbot_response:
                chatbot_response = "I'm sorry, but I'm currently unable to process your request."
                similarity_score = 0  # API failure → No valid comparison
            else:
                similarity_score = evaluate_similarity(chatbot_response, expected_answers)

        except Exception as e:
            print(f"❌ API Error: {e}")
            chatbot_response = "I'm sorry, but I'm currently unable to process your request."
            similarity_score = 0  # API failure → No valid comparison

        print(f"📌 Test Question: {test_question}")
        print(f"🤖 Chatbot Response: {chatbot_response}")
        print(f"📊 Similarity Score: {similarity_score:.4f}")

        results[intent] = {
            "question": test_question,
            "response": chatbot_response,
            "similarity": similarity_score
        }

    print("\n✅ Evaluation Complete!")
    return results, test_data

# ✅ Evaluate chatbot results
def evaluate_results(results, test_data):
    print("\n📈 **Final Evaluation Summary**")
    
    avg_similarity = np.mean([r["similarity"] for r in results.values()])
    
    for intent, data in results.items():
        print(f"📝 Intent: {intent}")
        print(f"📌 Query: {data['question']}")
        print(f"🤖 Chatbot Response: {data['response']}")
        print(f"📊 Similarity Score: {data['similarity']:.4f}\n")

    print(f"🏆 **Overall Avg Similarity:** {avg_similarity:.4f}")

# ✅ Main execution function
async def main():
    results, test_data = await evaluate_chatbot()
    evaluate_results(results, test_data)

if __name__ == "__main__":
    asyncio.run(main())
