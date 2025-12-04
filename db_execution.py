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



# Define a simple chain template for easy switching between models
def create_roles_chain(model):
    prompt = ChatPromptTemplate.from_template("{input}")
    return prompt | model | StrOutputParser() # prompt -> model -> StrOutputParser()

def generate_sql(question: str) -> str:
    schema_description = """
    Table: customers
    - id (integer)
    - name (text)
    - email (text)
    - signup_date (date)
    """

    final_prompt = f"""
    ### You are an expert SQL query writer. Who translates user requirements into sql queries. I have given you database schema.
    Use this database schema to generate sql queries.

    ### Database Schema:
    {schema_description}

    ### Question:
    {question}

    ### Output format: The output format should be json object with 'SQL' and value as  generated sql query
          "SQL": "..."

    """

    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite-001",
        google_api_key=gemini_api_key,
        temperature=0.3 ) 

    print("---   Response ---")

    sql_query = create_roles_chain(gemini_model).invoke({"input": final_prompt})
    match = re.search(r"```json\s*(\{.*?\})\s*```", sql_query, re.DOTALL)
    if not match:
        raise ValueError("No JSON code block found.")
    json_str = match.group(1)

    # Step 2: Parse JSON and extract the SQL
    data = json.loads(json_str)
    sql_query = data.get("SQL")
    return sql_query
