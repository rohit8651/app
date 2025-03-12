import faiss
import pickle
import os
import numpy as np
import requests
import time
from config import API_KEY, ENDPOINT_URL, RELEVANCE_THRESHOLD
from sentence_transformers import SentenceTransformer

# Load FAISS index
try:
    index = faiss.read_index("Data\\vector_database.index")
except Exception as e:
    raise FileNotFoundError("Error loading FAISS index: vector_database.index not found") from e

# Load file names & text chunks
try:
    with open("Data\\file_names.pkl", "rb") as f:
        file_names = pickle.load(f)

    with open("Data\\text_chunks.pkl", "rb") as f:
        text_chunks = pickle.load(f)
except FileNotFoundError as e:
    raise FileNotFoundError("Error loading file names or text chunks") from e

# Load the embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Optimized FAISS Search Function
def search_faiss(query, top_k=3):
    query_vector = embedding_model.encode(query, normalize_embeddings=True)  # Normalized embeddings
    query_vector = np.expand_dims(query_vector, axis=0)

    distances, indices = index.search(query_vector, top_k)

    if distances[0][0] > RELEVANCE_THRESHOLD:
        return "Out of context", []

    retrieved_chunks = [text_chunks[idx] for idx in indices[0] if idx < len(text_chunks)]
    return "Relevant", retrieved_chunks

# Retry logic for API calls
def call_api_with_retries(url, headers, payload, max_retries=3, timeout=90):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Timeout occurred, retrying {attempt + 1}/{max_retries}...")
            time.sleep(2 ** attempt)  # Exponential backoff
        except requests.exceptions.RequestException as e:
            print(f"API Request failed: {e}")
            return None
    return None  # Return None after max retries

# AI API query function
def query_company_ai(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)

    payload = {
        "model": "everest",
        "messages": [
            {"role": "system", "content": "You are a helpful chatbot. Only use the given context."},
            {"role": "user", "content": f"Query: {query}\n\nGIVEN CONTEXT:\n{context}"}
        ],
        "temperature": 0.0,
        "stream": False,
        "max_tokens": 1524  # Increased token limit for full response
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    response_json = call_api_with_retries(ENDPOINT_URL, headers, payload)

    if response_json and 'choices' in response_json and response_json['choices']:
        return response_json['choices'][0]['message']['content']
    
    return "Sorry, I couldn't process your request. Please try again."

# Main function to handle queries
def ask_question(query, top_k=3):
    relevance, retrieved_chunks = search_faiss(query, top_k)

    if relevance == "Out of context":
        return "Sorry, the query is not relevant to the available data."

    return query_company_ai(query, retrieved_chunks)
