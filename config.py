# ============================================
# 📌 CONFIGURATION FILE FOR RAG CHATBOT
# ============================================
import os
from dotenv import load_dotenv

# ✅ Load environment variables before defining API settings
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("❌ API Key is missing! Set HF_API_KEY in your .env file.")

# ✅ API Configuration
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# -------------------------------------------------
# ✅ BM25 Parameters (Lexical Retrieval - Keyword Matching)
# -------------------------------------------------
BM25_K1 = 1.5  # Controls term frequency saturation
# 🔽 Lower → Reduces the impact of term frequency (less influence from repeated words)
# 🔼 Higher → Increases the impact of term frequency (common words weigh more)

BM25_B = 0.75  # Controls document length normalization
# 🔽 Lower → Shorter documents are preferred (less normalization)
# 🔼 Higher → Longer documents are preferred (more normalization)

BM25_CONFIDENCE_THRESHOLD = 4.0  # Confidence threshold for BM25 retrieval
# 🔽 Lower → More results considered relevant (increases recall, may reduce precision)
# 🔼 Higher → Fewer results considered relevant (increases precision, may reduce recall)

# -------------------------------------------------
# ✅ FAISS Parameters (Semantic Retrieval - Vector Search)
# -------------------------------------------------
FAISS_CONFIDENCE_THRESHOLD = 0.6  # Minimum similarity score for FAISS matches
# 🔽 Lower → Allows more loosely related results (higher recall, risk of irrelevant matches)
# 🔼 Higher → Only highly similar results retrieved (higher precision, may exclude useful answers)

FAISS_NEIGHBORS_K = 1  # Number of nearest neighbors retrieved from FAISS
# 🔽 Lower → Fewer options to choose from (faster but may miss relevant info)
# 🔼 Higher → More options to choose from (slower but improves result diversity)

# -------------------------------------------------
# ✅ Falcon-7B API Parameters (LLM Response Control)
# -------------------------------------------------
API_MAX_TOKENS = 256  # Maximum number of tokens in response
# 🔽 Lower → Shorter responses (concise but may lack details)
# 🔼 Higher → Longer responses (detailed but may generate unnecessary text)

API_TEMPERATURE = 0.7  # Controls randomness in response generation
# 🔽 Lower → More deterministic responses (safer but repetitive)
# 🔼 Higher → More creative responses (varied but may be inconsistent)

API_TOP_P = 0.9  # Nucleus sampling - restricts randomness by keeping top-p probable choices
# 🔽 Lower → More focused, deterministic responses (reduces randomness)
# 🔼 Higher → More diverse, creative responses (adds randomness)

API_STREAM = False  # Enable streaming responses for real-time generation
# 🔽 False → Returns the full response at once (safer, more stable)
# 🔼 True → Streams the response token by token (useful for live chat UI)

# -------------------------------------------------
# ✅ Chatbot Branding & Support Contact
# -------------------------------------------------
COMPANY_NAME = "Company A"  # Company name for chatbot responses
SUPPORT_CONTACT = "+0123456789"  # Support contact for out-of-scope queries

CUSTOMER_SUPPORT_HOURS = "Monday to Friday, 9 AM - 6 PM"
CUSTOMER_SUPPORT_PHONE = "+0123456789"
WEBSITE_URL = "https://company-a.com/support"
LOGIN_PAGE_URL = "https://company-a.com/login"

# -------------------------------------------------
# ✅ Placeholder Mappings (Replace Specific Details with General Terms)
# -------------------------------------------------
PLACEHOLDER_REPLACEMENTS = {
    "{{Order Number}}": "your order details",
    "{{Invoice Number}}": "your invoice information",
    "{{Person Name}}": "the customer",
    "{{Account Type}}": "your account type",
    "{{Account Category}}": "your account category",
    "{{Delivery City}}": "your delivery location",
    "{{Delivery Country}}": "your country",
    "{{Currency Symbol}}": "the currency used",
    "{{Refund Amount}}": "your refund information",
    "{{Login Page URL}}": LOGIN_PAGE_URL,  # Dynamically insert login page URL
    "{{Forgot Key}}": "Forgot Key"
}

# -------------------------------------------------
# ✅ List of Placeholders to Remove Completely
# -------------------------------------------------
REMOVE_PLACEHOLDERS = [
    "Customer Support Phone Number",
    "Customer Support Email",
    "Claims Contact Number",
    "Order/Transaction ID",
    "Reference Number",
    "Tracking Number",
    "Account Number",
    "Customer ID",
    "Company Representative Name"
]

# -------------------------------------------------
# ✅ Custom Chat Prompt Template
# -------------------------------------------------
PROMPT_TEMPLATE = f"""
You are a customer support AI for {COMPANY_NAME}.
Here is the user's question:

User: "{{user_query}}"

We retrieved the following FAQ content (or fallback message) which might help:
{{retrieved_answer}}

Instructions:
- If the text above states it's out of scope, tell the user to contact {COMPANY_NAME}'s support at {SUPPORT_CONTACT}.
- Otherwise, incorporate the FAQ info in your own words (avoid copying verbatim).
- Maintain a polite, professional tone.

Now provide a final answer to the user:
"""
