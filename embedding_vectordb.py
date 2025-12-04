# !pip install chromadb sentence-transformers  langchain-google-genai==2.0.4

import os
import json
import numpy as np
from chromadb.utils import embedding_functions

# LangChain for Gemini Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# ChromaDB Client
import chromadb
from google.colab import userdata

# --- API Key Setup ---
# Use Colab Secrets for GEMINI_API_KEY
gemini_api_key = userdata.get("GOOGLE_API_KEY")


if not gemini_api_key:
    print("Warning: GEMINI_API_KEY not found. Gemini embedding will not run.")
else:
    print("Setup complete. Environment configured.")

# 2.1. Defining Sample Data: We will use four sample documents, two of which are conceptually similar ("Biking" and "Cycling").
documents = [
    "The official university policy states that all faculty must submit expense reports by the 15th of every month.", # Doc 1: Finance
    "Riding a bicycle provides excellent low-impact cardiovascular exercise and is a great way to commute.",           # Doc 2: Cycling
    "I enjoy going cycling on the weekends, especially when the weather is clear and the trails are dry.",           # Doc 3: Biking
    "Please consult the academic handbook regarding grading policies and attendance requirements for final year students." # Doc 4: Academics
]

# The user's query we want to compare against
user_query = "What is the best form of exercise using wheels?"
