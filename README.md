# 🏥 CliniqAI

An intelligent clinical assistant that unifies fragmented healthcare data into a single natural language interface. Built on real de-identified ICU data from MIMIC-III.

## The Problem

Health systems run on dozens of disconnected tools — patient records in one system, diagnoses in another, prescriptions somewhere else. Clinical staff waste hours hunting for information that should take seconds to find.

CliniqAI solves that.

## Demo

Ask anything about your patients in plain English:

- *"How many patients were admitted as emergency?"*
- *"Show me patients with cardiac conditions"*
- *"Show me cardiac patients and what they were prescribed"*
- *"Which patients died during admission?"*
- *"What medications are most commonly prescribed?"*

## Architecture

```
User Question
      ↓
Single-Token LLM Router (S / R / H / D)
      ↓
S → SQL only          — counting, aggregating, exact filters
R → RAG only          — semantic concepts, medical conditions
H → HYBRID            — parallel SQL + FAISS retrieval
D → DIRECT            — conversational, off-topic
      ↓
Groq + Llama 3.3 70B synthesizes grounded answer
      ↓
Answer displayed with route badge + SQL query
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Database | SQLite |
| Frontend | Streamlit |
| Data | MIMIC-III Clinical Database Demo |

## Data

Built on the [MIMIC-III Clinical Database Demo](https://physionet.org/content/mimiciii-demo/1.4/) — 100 de-identified ICU patients from Beth Israel Deaconess Medical Center.

- 100 patients
- 129 admissions
- 1,761 diagnoses
- 10,398 prescriptions

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Mynk711/Cliniq-AI.git
cd Cliniq-AI
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add Groq API key**
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

Get a free API key at [console.groq.com](https://console.groq.com)

**5. Download MIMIC-III Demo**

Download from [PhysioNet](https://physionet.org/content/mimiciii-demo/1.4/) and place in:
```
Cliniq-AI/mimic-iii-clinical-database-demo-1.4/
```

**6. Build the database and index**
```bash
python database.py
python document.py
```

**7. Run the app**
```bash
streamlit run app.py
```

## How It Works

### Router
Every question goes through a single-token LLM router that classifies intent into S/R/H/D in under 100ms — no keyword matching, no brittle rules.

### SQL Path
The LLM generates SQLite queries dynamically based on the schema and real column values. Aliases all aggregations for clean context passing.

### RAG Path
Patient records are converted to text chunks, embedded with sentence-transformers, and stored in a FAISS index. Semantic similarity search retrieves the most relevant patients.

### HYBRID Path
SQL and RAG run in parallel using ThreadPoolExecutor. Results are merged and passed to the LLM as combined context — structured data + semantic understanding in one answer.

### Responsible AI
- Answers grounded in real retrieved data only
- System prompt explicitly prevents hallucination
- Every query traceable via route badge and SQL expander
- Patient privacy maintained — referenced by ID only

## Limitations

- MIMIC-III demo dataset is limited to 100 patients
- Drug class semantic search requires exact generic names
- No authentication layer in current version
- Date shifting in MIMIC means all dates are offset for privacy

## What's Next

- Schema enrichment for smarter SQL generation
- Full MIMIC-III dataset (40K+ patients)
- Multi-tenant access controls
- Streaming responses
- Audit logging per user session

## Built By

Mayank Sehrawat — MS in Applied Machine Intelligence, Northeastern University

[LinkedIn](https://www.linkedin.com/in/mayanksehrawat711/) | [GitHub](https://github.com/Mynk711)