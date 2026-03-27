import gradio as gr
import torch
import numpy as np
import faiss
import pickle
import requests
import os
from dotenv import load_dotenv
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import warnings

# Disable warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# =========================================================================
# ⚙️ CLOUDFLARE WORKERS AI CONFIGURATION
# =========================================================================
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_API_TOKEN =  os.getenv("CF_API_TOKEN") 
CF_MODEL_ID =   os.getenv("CF_MODEL_ID", "@cf/meta/llama-3-8b-instruct")
# =========================================================================

# Initialize global data (runs once when the server starts)
print("Loading FAISS Database...")
try:
    index = faiss.read_index('company_policies.faiss')
    with open('paragraphs_list.pkl', 'rb') as f:
        paragraphs = pickle.load(f)
except FileNotFoundError:
    print("❌ Database not found. Please run 'python build_db.py' once before running this.")

print("Loading DPR Question Encoder (Text Search)...")
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

def search_relevant_contexts(question, k=3):
    question_inputs = question_tokenizer(question, return_tensors='pt')
    with torch.no_grad():
        question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()
    D, I = index.search(question_embedding, k)
    return D, I

def chatbot_response(message, history):
    """
    Communicates with Gradio Interface
    - Generates the answer and appends extracted citations for the UI
    """
    # 1. Retrieve Context
    distances, indices = search_relevant_contexts(message, k=3)
    retrieved_contexts = [paragraphs[idx] for idx in indices[0]]
    
    # 2. Generate answer via Cloudflare
    if not CF_ACCOUNT_ID or not CF_API_TOKEN or "dán_account_id" in CF_ACCOUNT_ID:
        return "⚠️ ERROR: You haven't correctly set Account ID or API Token in the .env file!"

    context_str = "\n".join(retrieved_contexts)
    prompt = f"Context information from company handbook:\n{context_str}\n\nGiven the context above, strictly answer the question: {message}"

    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL_ID}"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [
            {"role": "system", "content": "You are a professional HR assistant. Answer accurately based ONLY on the provided context. Make the answer natural, concise and directly related to the question."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        answer = result['result']['response'].strip()
    except Exception as e:
        answer = f"⚠️ Cloudflare API Connection Error: {str(e)}"
    
    # Append citations (References) at the end of the response
    citations = "\n\n---\n**📚 Extracted from Internal Documents:**\n"
    for ctx in retrieved_contexts:
        citations += f"- {ctx}\n"
        
    return answer + citations

# Build Web Interface
demo = gr.ChatInterface(
    fn=chatbot_response,                 # Call logic on every chat message
    title="🏢 Company RAG Chatbot",
    description="Ask any question regarding our company policies. Powered by FAISS & Cloudflare Llama-3.",
    examples=[
        "Tell me about the Mobile Phone Policy",
        "What is the Code of Conduct?",
        "When will employees be notified about termination?"
    ]
)

if __name__ == "__main__":
    print("\n✅ Web Server is starting...")
    demo.launch()
