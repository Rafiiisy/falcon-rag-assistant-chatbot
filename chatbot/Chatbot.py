# ============================================
# 📌 CLI-BASED CHATBOT INTERFACE
# ============================================

import asyncio
from inference import chat_with_falcon_api

async def chat_loop():
    """Runs an interactive chatbot session."""
    print("\n💬 Type your message below. Type 'exit' to stop.\n")

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() == "exit":
            print("🔌 Chatbot session ended.")
            break
        response = await chat_with_falcon_api(user_input)
        print(f"🤖 Falcon-7B: {response}")

if __name__ == '__main__':
    asyncio.run(chat_loop())
