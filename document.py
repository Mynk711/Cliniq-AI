import sqlite3
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

DB_PATH = "cliniqai.db"
INDEX_PATH = "cliniqai.index"
CHUNKS_PATH = "cliniqai_chunks.pkl"

# Load embedding model — runs locally, no API needed
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_patient_chunks():
    """Convert real MIMIC records into text chunks for embedding"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            p.subject_id,
            p.gender,
            p.dob,
            p.dod,
            a.hadm_id,
            a.admittime,
            a.dischtime,
            a.admission_type,
            a.diagnosis,
            d.icd9_code,
            pr.drug,
            pr.dose_val_rx,
            pr.dose_unit_rx
        FROM patients p
        LEFT JOIN admissions a ON p.subject_id = a.subject_id
        LEFT JOIN diagnoses d ON a.hadm_id = d.hadm_id
        LEFT JOIN prescriptions pr ON a.hadm_id = pr.hadm_id
        WHERE p.subject_id IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    # Group by patient
    patients = {}
    for row in rows:
        sid = row[0]
        if sid not in patients:
            patients[sid] = {
                "subject_id": row[0],
                "gender": row[1],
                "dob": row[2],
                "dod": row[3],
                "admissions": [],
                "diagnoses": [],
                "prescriptions": []
            }
        if row[4]:  # hadm_id
            admission = f"Admitted: {row[5]}, Discharged: {row[6]}, Type: {row[7]}, Diagnosis: {row[8]}"
            if admission not in patients[sid]["admissions"]:
                patients[sid]["admissions"].append(admission)
        if row[9]:  # icd9_code
            if row[9] not in patients[sid]["diagnoses"]:
                patients[sid]["diagnoses"].append(row[9])
        if row[10]:  # drug
            prescription = f"{row[10]} {row[11]}{row[12]}"
            if prescription not in patients[sid]["prescriptions"]:
                patients[sid]["prescriptions"].append(prescription)

    # Convert each patient to a text chunk
    chunks = []
    for sid, data in patients.items():
        text = f"""
Patient ID: {data['subject_id']}
Gender: {data['gender']}
Date of Birth: {data['dob']}
Date of Death: {data['dod'] if data['dod'] else 'N/A'}

Admission History:
{chr(10).join(data['admissions']) if data['admissions'] else 'No admissions recorded'}

Diagnoses (ICD9):
{', '.join(data['diagnoses']) if data['diagnoses'] else 'No diagnoses recorded'}

Prescriptions:
{chr(10).join(data['prescriptions']) if data['prescriptions'] else 'No prescriptions recorded'}
        """.strip()

        chunks.append({
            "subject_id": sid,
            "text": text
        })

    return chunks

def build_index():
    """Embed chunks and store in FAISS index"""
    print("Building patient chunks from MIMIC data...")
    chunks = build_patient_chunks()
    print(f"Created {len(chunks)} patient chunks")

    print("Generating embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index and chunks
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index built with {index.ntotal} vectors")
    print("Saved to cliniqai.index and cliniqai_chunks.pkl")

def retrieve(query, top_k=5):
    """Retrieve most relevant patient chunks for a query"""
    # Load index and chunks
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # Embed query
    query_embedding = model.encode([query]).astype("float32")

    # Search
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "subject_id": chunks[idx]["subject_id"],
                "text": chunks[idx]["text"],
                "score": float(distances[0][i])
            })

    return results

if __name__ == "__main__":
    build_index()