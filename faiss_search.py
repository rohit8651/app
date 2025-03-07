import faiss
import pickle
import os
import numpy as np
import requests
from config import API_KEY, ENDPOINT_URL, RELEVANCE_THRESHOLD
from sentence_transformers import SentenceTransformer
# from openai import OpenAI
import google.generativeai as genai

# Load FAISS index
try:
    index = faiss.read_index("vector_database.index")
except Exception as e:
    raise FileNotFoundError("Error loading FAISS index: vector_database.index not found") from e

# Load file names & text chunks
try:
    with open("file_names.pkl", "rb") as f:
        file_names = pickle.load(f)

    with open("text_chunks.pkl", "rb") as f:
        text_chunks = pickle.load(f)
except FileNotFoundError as e:
    raise FileNotFoundError("Error loading file names or text chunks") from e

# Load the embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Search function
def search_faiss(query, top_k=3):
    query_vector = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy()
    query_vector = np.expand_dims(query_vector, axis=0)

    distances, indices = index.search(query_vector, top_k)

    if distances[0][0] > RELEVANCE_THRESHOLD:
        return "Out of context", []

    retrieved_chunks = [text_chunks[idx] for idx in indices[0]]
    return "Relevant", retrieved_chunks

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
        "max_tokens": 512
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        response_json = response.json()

        if 'choices' in response_json and response_json['choices']:
            return response_json['choices'][0]['message']['content']
        else:
            return "Error: Unexpected API response."

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"



# Main function to handle queries
def ask_question(query, top_k=3):
    relevance, retrieved_chunks = search_faiss(query, top_k)

    if relevance == "Out of context":
        return "Sorry, the query is not relevant to the available data."

    return query_company_ai(query, retrieved_chunks)
