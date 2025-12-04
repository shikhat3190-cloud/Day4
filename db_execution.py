import os, re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.colab import userdata
import sqlite3
import json

# Get API keys from environment variables
gemini_api_key = userdata.get("GOOGLE_API_KEY")
if not gemini_api_key:
    print("Warning: GEMINI_API_KEY not found. Gemini model will not run.")


def setupdb():
    conn = sqlite3.connect("example.db") # 
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        signup_date DATE
    )
    """)

    # Insert some sample data
    cursor.executemany("""
    INSERT INTO customers (id,name, email, signup_date)
    VALUES (?,?, ?, ?)
    """, [
        (1, "Alice Johnson", "alice@example.com", "2025-06-15"),
        (2, "Bob Smith", "bob@example.com", "2024-07-01"),
        (3, "Clara White", "clara@example.com", "2024-07-10")
    ])

    conn.commit()
    conn.close()
