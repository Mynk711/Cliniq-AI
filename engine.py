import warnings
warnings.filterwarnings("ignore")

import os
import re
from groq import Groq
from document import retrieve
from database import query_db
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, END

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

Important: This data is from MIMIC-III, a de-identified clinical database. All dates have been shifted forward by a random offset per patient to protect privacy — this is why dates may appear to be in the future. Relative time intervals between events are preserved.

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

# ── LangGraph State ───────────────────────────────────────────────────────────
class CliniqState(TypedDict):
    question: str
    chat_history: List[dict]
    route: str
    columns: Optional[Any]
    rows: Optional[Any]
    sql: Optional[str]
    chunks: Optional[Any]
    context: str
    final_answer: str
    sql_used: Optional[str]

# ── Core functions (unchanged) ────────────────────────────────────────────────
def classify_query(question):
    if re.search(r'\b\d{4,6}\b', question):
        return "s"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": ROUTER_PROMPT.format(question=question)}],
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
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
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
    data_rows = "\n".join([" | ".join(str(v) for v in row) for row in display])
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
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.3)
    return response.choices[0].message.content.strip()

def answer_direct(question, chat_history=[]):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.5)
    return response.choices[0].message.content.strip()

# ── LangGraph Nodes ───────────────────────────────────────────────────────────
def router_node(state: CliniqState) -> dict:
    route = classify_query(state["question"])
    return {"route": route}

def sql_node(state: CliniqState) -> dict:
    columns, rows, sql = run_sql_pipeline(state["question"], state["chat_history"])
    context = format_sql_context(columns, rows)
    return {"columns": columns, "rows": rows, "sql": sql, "context": context, "sql_used": sql}

def rag_node(state: CliniqState) -> dict:
    chunks = run_rag_pipeline(state["question"])
    context = format_rag_context(chunks)
    return {"chunks": chunks, "context": context, "sql_used": None}

def hybrid_node(state: CliniqState) -> dict:
    with ThreadPoolExecutor(max_workers=2) as executor:
        sql_future = executor.submit(run_sql_pipeline, state["question"], state["chat_history"])
        rag_future = executor.submit(run_rag_pipeline, state["question"])
        columns, rows, sql = sql_future.result()
        chunks = rag_future.result()
    sql_context = format_sql_context(columns, rows)
    rag_context = format_rag_context(chunks)
    combined = f"STRUCTURED DATABASE RESULTS:\n{sql_context}\n\nSEMANTIC PATIENT RECORDS:\n{rag_context}"
    return {"columns": columns, "rows": rows, "sql": sql, "chunks": chunks, "context": combined, "sql_used": sql}

def direct_node(state: CliniqState) -> dict:
    ans = answer_direct(state["question"], state["chat_history"])
    return {"final_answer": ans, "sql_used": None}

def synthesize_node(state: CliniqState) -> dict:
    ans = synthesize(state["question"], state["context"], state["chat_history"])
    return {"final_answer": ans}

def route_decision(state: CliniqState) -> str:
    return state["route"]

# ── Build Graph ───────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(CliniqState)

    graph.add_node("router",     router_node)
    graph.add_node("sql",        sql_node)
    graph.add_node("rag",        rag_node)
    graph.add_node("hybrid",     hybrid_node)
    graph.add_node("direct",     direct_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "s": "sql",
        "r": "rag",
        "h": "hybrid",
        "d": "direct"
    })
    graph.add_edge("sql",    "synthesize")
    graph.add_edge("rag",    "synthesize")
    graph.add_edge("hybrid", "synthesize")
    graph.add_edge("direct", END)
    graph.add_edge("synthesize", END)

    return graph.compile()

_graph = build_graph()

# ── Main entry point (same signature as before) ───────────────────────────────
def answer(question, chat_history=[]):
    result = _graph.invoke({
        "question":     question,
        "chat_history": chat_history,
        "route":        "",
        "columns":      None,
        "rows":         None,
        "sql":          None,
        "chunks":       None,
        "context":      "",
        "final_answer": "",
        "sql_used":     None
    })

    route = result["route"].upper()
    method_map = {"S": "SQL", "R": "RAG", "H": "HYBRID", "D": "DIRECT"}

    return {
        "answer":   result["final_answer"],
        "method":   method_map.get(route, "DIRECT"),
        "sql_used": result.get("sql_used"),
        "route":    route
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

# import warnings
# warnings.filterwarnings("ignore")

# import os
# import re
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

# Important: This data is from MIMIC-III, a de-identified clinical database. All dates have been shifted forward by a random offset per patient to protect privacy - this is why dates may appear to be in the future. Relative time intervals between events are preserved.

# Your job is to answer questions accurately and concisely using ONLY the context provided.
# If the context doesn't contain enough information to answer, say so clearly.
# Never make up patient information.
# Always maintain patient privacy - refer to patients by ID only.
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
#     # If a specific patient ID is mentioned, always use SQL
#     if re.search(r'\b\d{4,6}\b', question):
#         return "s"

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
#     return retrieve(question, top_k=3)

# def format_sql_context(columns, rows):
#     if columns is None or not rows:
#         return "No records found."
#     total = len(rows)
#     display = rows[:50]
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