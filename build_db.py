import wget
import os
import torch
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import faiss
import pickle
import warnings

# Disable warnings for a cleaner terminal output
warnings.filterwarnings('ignore')

def read_and_split_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    # Split the text into separate paragraphs
    paragraphs = text.split('\n')
    # Filter out empty or whitespace-only paragraphs
    paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 0]
    return paragraphs

def main():
    filename = 'companyPolicies.txt'
    url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'
    
    # 1. Download the document if it doesn't exist
    if not os.path.exists(filename):
        print("[1/5] Downloading company document dataset...")
        wget.download(url, out=filename)
        print("\nDownload complete.")
    else:
        print("[1/5] Document file already exists.")

    # 2. Read and chunk the data
    print("[2/5] Reading and splitting text into paragraphs...")
    paragraphs = read_and_split_text(filename)
    
    # 3. Load the Context Encoder model
    print("[3/5] Loading Context Encoder model (this might take a few seconds)...")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    # 4. Start encoding the documents
    print("[4/5] Encoding contexts (Converting text into numerical vectors)...")
    embeddings = []
    for text in paragraphs:
        inputs = context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        # Use torch.no_grad() to save RAM during inference
        with torch.no_grad():
            outputs = context_encoder(**inputs)
        embeddings.append(outputs.pooler_output)
    
    context_embeddings = torch.cat(embeddings).detach().numpy()
    
    # 5. Initialize and save the Database Index using FAISS
    print("[5/5] Building and saving Vector Database (FAISS Index)...")
    embedding_dim = 768
    context_embeddings_np = np.array(context_embeddings).astype('float32')
    
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(context_embeddings_np)
    
    # Export Index and Paragraph list to local storage
    faiss.write_index(index, 'company_policies.faiss')
    with open('paragraphs_list.pkl', 'wb') as f:
        pickle.dump(paragraphs, f)
        
    print("\n✅ SUCCESS! Vector Database built and saved safely. You can now close this and run chat.py.")

if __name__ == "__main__":
    main()
