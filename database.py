import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "cliniqai.db")
MIMIC_PATH = "mimic-iii-clinical-database-demo-1.4"

def init_db():
    conn = sqlite3.connect(DB_PATH)

    # Load PATIENTS — first 100
    patients = pd.read_csv(f"{MIMIC_PATH}/PATIENTS.csv")
    patients = patients[["subject_id", "gender", "dob", "dod"]].head(100)
    patients.to_sql("patients", conn, if_exists="replace", index=False)

    # Get our patient IDs
    patient_ids = patients["subject_id"].tolist()
    ids_str = ",".join(map(str, patient_ids))

    # Load ADMISSIONS — only for our patients
    admissions = pd.read_csv(f"{MIMIC_PATH}/ADMISSIONS.csv")
    admissions = admissions[admissions["subject_id"].isin(patient_ids)]
    admissions = admissions[[
        "subject_id", "hadm_id", "admittime",
        "dischtime", "admission_type", "diagnosis"
    ]]
    admissions.to_sql("admissions", conn, if_exists="replace", index=False)

    # Get our hadm_ids
    hadm_ids = admissions["hadm_id"].tolist()

    # Load DIAGNOSES — only for our hadm_ids
    diagnoses = pd.read_csv(f"{MIMIC_PATH}/DIAGNOSES_ICD.csv")
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(hadm_ids)]
    diagnoses = diagnoses[["subject_id", "hadm_id", "icd9_code"]]
    diagnoses.to_sql("diagnoses", conn, if_exists="replace", index=False)

    # Load PRESCRIPTIONS — only for our hadm_ids
    prescriptions = pd.read_csv(f"{MIMIC_PATH}/PRESCRIPTIONS.csv")
    prescriptions = prescriptions[prescriptions["hadm_id"].isin(hadm_ids)]
    prescriptions = prescriptions[[
        "subject_id", "hadm_id", "drug",
        "dose_val_rx", "dose_unit_rx"
    ]]
    prescriptions.to_sql("prescriptions", conn, if_exists="replace", index=False)

    # Add indexes for faster queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_admissions_subject ON admissions(subject_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_admissions_hadm ON admissions(hadm_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_diagnoses_hadm ON diagnoses(hadm_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_hadm ON prescriptions(hadm_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_drug ON prescriptions(drug)")
    
    conn.commit()
    conn.close()
    print("Database initialized with MIMIC data")
    print(f"Patients: {len(patients)}")
    print(f"Admissions: {len(admissions)}")
    print(f"Diagnoses: {len(diagnoses)}")
    print(f"Prescriptions: {len(prescriptions)}")
    

def query_db(sql):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return columns, rows
    except Exception as e:
        conn.close()
        return None, str(e)

if __name__ == "__main__":
    init_db()