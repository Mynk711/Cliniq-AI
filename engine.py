# import warnings
# warnings.filterwarnings("ignore")

# import os
# from groq import Groq
# from document import retrieve
# from database import query_db
# from dotenv import load_dotenv
# from concurrent.futures import ThreadPoolExecutor

# load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# MODEL = "llama-3.3-70b-versatile"

# SYSTEM_PROMPT = """
# You are CliniqAI, an intelligent clinical assistant for healthcare operators.
# You have access to real patient data from a hospital database including:
# - Patient demographics
# - Admission history
# - Diagnoses (ICD9 codes)
# - Prescriptions

# Your job is to answer questions accurately and concisely using ONLY the context provided.
# If the context doesn't contain enough information to answer, say so clearly.
# Never make up patient information.
# Always maintain patient privacy — refer to patients by ID only.
# Format responses cleanly and clearly for clinical staff.
# """

# ROUTER_PROMPT = """
# You are a clinical query router. Classify the user query into exactly one category.

# S = SQL only: counting, aggregating, exact filters, dates, rankings, numerical analysis
# R = RAG only: semantic concepts, medical conditions, patient summaries, open-ended clinical questions  
# H = HYBRID: requires both semantic search AND structured data (e.g. "cardiac patients and their prescriptions")
# D = DIRECT: greetings, off-topic, out of scope, non-clinical questions

# Respond with EXACTLY one character: S, R, H, or D
# No punctuation. No explanation. One character only.

# Query: {question}
# """

# def classify_query(question):
#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[{
#             "role": "user",
#             "content": ROUTER_PROMPT.format(question=question)
#         }],
#         max_tokens=1,
#         temperature=0
#     )
#     decision = response.choices[0].message.content.strip().upper()
#     if decision not in ["S", "R", "H", "D"]:
#         return "d"
#     return decision.lower()

# def generate_sql(question, chat_history=[]):
#     schema = """
#     Tables:
#     - patients: subject_id, gender, dob, dod
#     - admissions: subject_id, hadm_id, admittime, dischtime, admission_type, diagnosis
#     - diagnoses: subject_id, hadm_id, icd9_code
#     - prescriptions: subject_id, hadm_id, drug, dose_val_rx, dose_unit_rx

#     Relationships:
#     - patients.subject_id = admissions.subject_id
#     - admissions.hadm_id = diagnoses.hadm_id
#     - admissions.hadm_id = prescriptions.hadm_id

#     Important notes:
#     - admission_type values are exactly: EMERGENCY, ELECTIVE, URGENT
#     - All drug names are in title case
#     - Cardiac ICD9 codes start with 410-429
#     - Beta blockers are stored as: Metoprolol Tartrate, Metoprolol Succinate XL,
#       Carvedilol, Atenolol, Labetalol, Propranolol
#     - When searching drugs use LIKE with exact casing e.g. drug LIKE 'Metoprolol%'
#     """

#     prompt = f"""
#     Given this database schema:
#     {schema}

#     Generate a valid SQLite SQL query to answer this question:
#     "{question}"

#     Rules:
#     - Always alias COUNT columns descriptively e.g. COUNT(*) AS total_patients
#     - Always alias aggregation columns e.g. AVG(age) AS average_age
#     - Return ONLY the SQL query, nothing else.
#     - No explanation, no markdown, no backticks.
#     """

#     messages = [{"role": "system", "content": "You are a SQL expert for a healthcare SQLite database."}]
#     messages.extend(chat_history)
#     messages.append({"role": "user", "content": prompt})

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=messages,
#         temperature=0
#     )
#     return response.choices[0].message.content.strip()

# def run_sql_pipeline(question, chat_history=[]):
#     sql = generate_sql(question, chat_history)
#     columns, rows = query_db(sql)
#     return columns, rows, sql

# def run_rag_pipeline(question):
#     return retrieve(question, top_k=4)

# def format_sql_context(columns, rows):
#     if columns is None or not rows:
#         return "No records found."
#     total = len(rows)
#     display = rows[:100]
#     header = " | ".join(columns)
#     data_rows = "\n".join([
#         " | ".join(str(v) for v in row) for row in display
#     ])
#     suffix = f"\n(Showing {len(display)} of {total} total results)" if total > len(display) else f"\n(Total: {total} results)"
#     return f"Query returned {total} results:\n{header}\n{data_rows}{suffix}"

# def format_rag_context(chunks):
#     if not chunks:
#         return "No relevant patient records found."
#     return "\n\n---\n\n".join([c["text"] for c in chunks])

# def synthesize(question, context, chat_history=[]):
#     prompt = f"""
#     Question: {question}

#     Clinical data context:
#     {context}

#     Answer the question clearly and concisely based only on the provided context.
#     """

#     messages = [{"role": "system", "content": SYSTEM_PROMPT}]
#     messages.extend(chat_history)
#     messages.append({"role": "user", "content": prompt})

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=messages,
#         temperature=0.3
#     )
#     return response.choices[0].message.content.strip()

# def answer_direct(question, chat_history=[]):
#     messages = [{"role": "system", "content": SYSTEM_PROMPT}]
#     messages.extend(chat_history)
#     messages.append({"role": "user", "content": question})

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=messages,
#         temperature=0.5
#     )
#     return response.choices[0].message.content.strip()

# def answer(question, chat_history=[]):
#     route = classify_query(question)

#     if route == "d":
#         return {
#             "answer": answer_direct(question, chat_history),
#             "method": "DIRECT",
#             "sql_used": None,
#             "route": "D"
#         }

#     if route == "s":
#         columns, rows, sql = run_sql_pipeline(question, chat_history)
#         context = format_sql_context(columns, rows)
#         return {
#             "answer": synthesize(question, context, chat_history),
#             "method": "SQL",
#             "sql_used": sql,
#             "route": "S"
#         }

#     if route == "r":
#         chunks = run_rag_pipeline(question)
#         context = format_rag_context(chunks)
#         return {
#             "answer": synthesize(question, context, chat_history),
#             "method": "RAG",
#             "sql_used": None,
#             "route": "R"
#         }

#     if route == "h":
#         with ThreadPoolExecutor(max_workers=2) as executor:
#             sql_future = executor.submit(run_sql_pipeline, question, chat_history)
#             rag_future = executor.submit(run_rag_pipeline, question)
#             columns, rows, sql = sql_future.result()
#             chunks = rag_future.result()

#         sql_context = format_sql_context(columns, rows)
#         rag_context = format_rag_context(chunks)

#         combined_context = f"""
# STRUCTURED DATABASE RESULTS:
# {sql_context}

# SEMANTIC PATIENT RECORDS:
# {rag_context}
#         """.strip()

#         return {
#             "answer": synthesize(question, combined_context, chat_history),
#             "method": "HYBRID",
#             "sql_used": sql,
#             "route": "H"
#         }

#     return {
#         "answer": answer_direct(question, chat_history),
#         "method": "DIRECT",
#         "sql_used": None,
#         "route": "D"
#     }

# if __name__ == "__main__":
#     test_questions = [
#         "How many patients were admitted as emergency?",
#         "Show me patients with cardiac conditions",
#         "Which cardiac patients were prescribed beta blockers?",
#         "What is your favorite color?",
#     ]

#     for q in test_questions:
#         print(f"\nQ: {q}")
#         result = answer(q)
#         print(f"Route: {result['route']} | Method: {result['method']}")
#         if result['sql_used']:
#             print(f"SQL: {result['sql_used']}")
#         print(f"A: {result['answer']}")
#         print("="*50)


import warnings
warnings.filterwarnings("ignore")

import os
import re
from groq import Groq
from document import retrieve
from database import query_db
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are CliniqAI, an intelligent clinical assistant for healthcare operators.
You have access to real patient data from a hospital database including:
- Patient demographics
- Admission history
- Diagnoses (ICD9 codes)
- Prescriptions

Your job is to answer questions accurately and concisely using ONLY the context provided.
If the context doesn't contain enough information to answer, say so clearly.
Never make up patient information.
Always maintain patient privacy — refer to patients by ID only.
Format responses cleanly and clearly for clinical staff.
"""

ROUTER_PROMPT = """
You are a clinical query router. Classify the user query into exactly one category.

S = SQL only: counting, aggregating, exact filters, dates, rankings, numerical analysis
R = RAG only: semantic concepts, medical conditions, patient summaries, open-ended clinical questions  
H = HYBRID: requires both semantic search AND structured data (e.g. "cardiac patients and their prescriptions")
D = DIRECT: greetings, off-topic, out of scope, non-clinical questions

Respond with EXACTLY one character: S, R, H, or D
No punctuation. No explanation. One character only.

Query: {question}
"""

def classify_query(question):
    # If a specific patient ID is mentioned, always use SQL
    if re.search(r'\b\d{4,6}\b', question):
        return "s"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": ROUTER_PROMPT.format(question=question)
        }],
        max_tokens=1,
        temperature=0
    )
    decision = response.choices[0].message.content.strip().upper()
    if decision not in ["S", "R", "H", "D"]:
        return "d"
    return decision.lower()

def generate_sql(question, chat_history=[]):
    schema = """
    Tables:
    - patients: subject_id, gender, dob, dod
    - admissions: subject_id, hadm_id, admittime, dischtime, admission_type, diagnosis
    - diagnoses: subject_id, hadm_id, icd9_code
    - prescriptions: subject_id, hadm_id, drug, dose_val_rx, dose_unit_rx

    Relationships:
    - patients.subject_id = admissions.subject_id
    - admissions.hadm_id = diagnoses.hadm_id
    - admissions.hadm_id = prescriptions.hadm_id

    Important notes:
    - admission_type values are exactly: EMERGENCY, ELECTIVE, URGENT
    - All drug names are in title case
    - Cardiac ICD9 codes start with 410-429
    - Beta blockers are stored as: Metoprolol Tartrate, Metoprolol Succinate XL,
      Carvedilol, Atenolol, Labetalol, Propranolol
    - When searching drugs use LIKE with exact casing e.g. drug LIKE 'Metoprolol%'
    """

    prompt = f"""
    Given this database schema:
    {schema}

    Generate a valid SQLite SQL query to answer this question:
    "{question}"

    Rules:
    - Always alias COUNT columns descriptively e.g. COUNT(*) AS total_patients
    - Always alias aggregation columns e.g. AVG(age) AS average_age
    - Return ONLY the SQL query, nothing else.
    - No explanation, no markdown, no backticks.
    """

    messages = [{"role": "system", "content": "You are a SQL expert for a healthcare SQLite database."}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def run_sql_pipeline(question, chat_history=[]):
    sql = generate_sql(question, chat_history)
    columns, rows = query_db(sql)
    return columns, rows, sql

def run_rag_pipeline(question):
    return retrieve(question, top_k=3)

def format_sql_context(columns, rows):
    if columns is None or not rows:
        return "No records found."
    total = len(rows)
    display = rows[:50]
    header = " | ".join(columns)
    data_rows = "\n".join([
        " | ".join(str(v) for v in row) for row in display
    ])
    suffix = f"\n(Showing {len(display)} of {total} total results)" if total > len(display) else f"\n(Total: {total} results)"
    return f"Query returned {total} results:\n{header}\n{data_rows}{suffix}"

def format_rag_context(chunks):
    if not chunks:
        return "No relevant patient records found."
    return "\n\n---\n\n".join([c["text"] for c in chunks])

def synthesize(question, context, chat_history=[]):
    prompt = f"""
    Question: {question}

    Clinical data context:
    {context}

    Answer the question clearly and concisely based only on the provided context.
    """

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def answer_direct(question, chat_history=[]):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def answer(question, chat_history=[]):
    route = classify_query(question)

    if route == "d":
        return {
            "answer": answer_direct(question, chat_history),
            "method": "DIRECT",
            "sql_used": None,
            "route": "D"
        }

    if route == "s":
        columns, rows, sql = run_sql_pipeline(question, chat_history)
        context = format_sql_context(columns, rows)
        return {
            "answer": synthesize(question, context, chat_history),
            "method": "SQL",
            "sql_used": sql,
            "route": "S"
        }

    if route == "r":
        chunks = run_rag_pipeline(question)
        context = format_rag_context(chunks)
        return {
            "answer": synthesize(question, context, chat_history),
            "method": "RAG",
            "sql_used": None,
            "route": "R"
        }

    if route == "h":
        with ThreadPoolExecutor(max_workers=2) as executor:
            sql_future = executor.submit(run_sql_pipeline, question, chat_history)
            rag_future = executor.submit(run_rag_pipeline, question)
            columns, rows, sql = sql_future.result()
            chunks = rag_future.result()

        sql_context = format_sql_context(columns, rows)
        rag_context = format_rag_context(chunks)

        combined_context = f"""
STRUCTURED DATABASE RESULTS:
{sql_context}

SEMANTIC PATIENT RECORDS:
{rag_context}
        """.strip()

        return {
            "answer": synthesize(question, combined_context, chat_history),
            "method": "HYBRID",
            "sql_used": sql,
            "route": "H"
        }

    return {
        "answer": answer_direct(question, chat_history),
        "method": "DIRECT",
        "sql_used": None,
        "route": "D"
    }

if __name__ == "__main__":
    test_questions = [
        "How many patients were admitted as emergency?",
        "Show me patients with cardiac conditions",
        "Which cardiac patients were prescribed beta blockers?",
        "What is your favorite color?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        result = answer(q)
        print(f"Route: {result['route']} | Method: {result['method']}")
        if result['sql_used']:
            print(f"SQL: {result['sql_used']}")
        print(f"A: {result['answer']}")
        print("="*50)