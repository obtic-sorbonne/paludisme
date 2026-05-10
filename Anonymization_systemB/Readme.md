# System B – CNR Paludisme Medical Digitization Pipeline

Fully automated pipeline: scanned PDFs → anonymized text → structured research table (Excel + SQLite).

---

## Quick Start

```bash
# One command does everything:
bash run_pipeline.sh "<path>"
```

The script **auto-detects** what you gave it:

| What you pass | What happens |
|---|---|
| Path to a single `.pdf` file | Processes that one PDF |
| Path to one patient folder | Processes all PDFs in that folder |
| Path to a folder containing 100s of patient folders | Processes every patient subfolder |

---

## Usage Examples

```bash
# ── Process one patient folder ─────────────────────────────────────────────
bash run_pipeline.sh \
  "/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0186/"

# ── Process ALL 15 patients in the default batch folder ────────────────────
bash run_pipeline.sh \
  "/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/"

# ── Process 800 patients in any folder ─────────────────────────────────────
bash run_pipeline.sh "/path/to/folder_with_800_patient_folders/"

# ── Re-run ONLY variable extraction for all existing anonymized files ───────
# Use this if a patient is missing from the database, or after editing the
# config file to fix/add variable patterns
python ~/digitize_medical_records/VariableExtraction/extract_variables.py \
  --all \
  --config ~/digitize_medical_records/VariableExtraction/variable_extraction_config.yaml \
  --output-dir ~/digitize_medical_records/VariableExtraction/outputs

# ── Run pipeline but skip Excel/SQLite output ───────────────────────────────
bash run_pipeline.sh "/path/to/patients/" --skip-variables
```

---

## What Happens Step by Step

```
INPUT: Patient folder (e.g. "2006 RDB 0186/")
  └── Contains: DOC_00118.pdf, DOC_00119.pdf, DOC_00122.pdf ...

STEP 1: GLM-OCR
  Each PDF is processed by GLM-OCR.
  Produces: raw text + page images per PDF.
  Output: /home/lfarooq/glm_ocr_runs/<doc_id>/

STEP 2: Page Classification (Qwen 7B vision model)
  Each page image is shown to Qwen 7B.
  It answers YES or NO: "Is this a CNR Paludisme form?"
  Produces: list of CNR pages and non-CNR pages per document.

STEP 3: Extraction
  ├── CNR pages    → Qwen 72B extracts all form fields
  │                  (species, dates, treatment, lab values, etc.)
  │                  Then postprocess_checker.py validates output.
  └── Non-CNR pages → GLM-OCR text only
                      (lab results, clinical reports, urgences)

STEP 4: Merge
  All document outputs merged into one file per patient:
  Output: outputs/patients/RDB_0186/RDB_0186_patient_raw.txt

STEP 5: Anonymization
  Reads patient_raw.txt.
  Removes all personal identifiers:
    - Patient name  → [PATIENT_001]
    - Doctor names  → [STAFF_001], [STAFF_002] ...
    - Phone, email, address, NPI, FINESS, séjour IDs → [ANONYMIZED]
  Uses Qwen 7B NER (local, no internet) to detect names in free text.
  Outputs:
    outputs/patients/RDB_0186/RDB_0186_patient_anonymized.txt
    outputs/patients/RDB_0186/RDB_0186_replacements.csv
    outputs/final_anonymized/patient_002_anonymized.txt  ← central copy

STEP 6: Variable Extraction
  Runs ONCE at the end of the batch (not per-patient).
  Reads ALL files in outputs/final_anonymized/.
  Assigns each document/page to J0, J3, or J30 by consultation date order:
    - J0  = first consultation date (admission)
    - J3  = next consultation date beyond 3 days from J0
    - J30 = last consultation date
    - Dates within 3 days of J0 are merged into J0
  Extracts 130 clinical variables:
    Demographics, travel, symptoms, vitals, lab values, treatment, outcome
  Clinical document values take priority over CNR form values.
  Writes one row per patient to:
    VariableExtraction/outputs/research_table.xlsx   ← Excel (all patients)
    VariableExtraction/outputs/research_database.db  ← SQLite (all patients)
```

---

## Output Files

```
digitize_medical_records/
├── outputs/
│   ├── patients/
│   │   ├── RDB_0186/
│   │   │   ├── RDB_0186_patient_raw.txt          ← merged raw text
│   │   │   ├── RDB_0186_patient_anonymized.txt   ← anonymized text
│   │   │   └── RDB_0186_replacements.csv         ← what was replaced
│   │   └── RDB_XXXX/ ...
│   └── final_anonymized/
│       ├── patient_001_anonymized.txt
│       ├── patient_002_anonymized.txt
│       └── ...
│
└── VariableExtraction/
    ├── extract_variables.py
    ├── variable_extraction_config.yaml   ← only file new institutions need to edit
    └── outputs/
        ├── research_table.xlsx           ← ONE Excel file, all patients (one row each)
        └── research_database.db         ← ONE SQLite database, all patients
```

> **Important:** Every new patient is **appended** to the same Excel file and
> the same SQLite database. Running the pipeline on 15 patients gives you
> 15 rows in one sheet. Running again on the same patient **updates** their row
> (does not duplicate).

---

## Viewing Results

### Excel
Open `research_table.xlsx` directly in VSCode using the **Excel Viewer** extension
(MESCIUS/GrapeCity) — click the file, then click "Open Anyway".

Or download to your local machine:
```bash
# Run this on your LOCAL machine terminal:
scp lfarooq@134.157.57.238:~/digitize_medical_records/VariableExtraction/outputs/research_table.xlsx ~/Desktop/
```

### SQLite
Open `research_database.db` in VSCode using the **SQLite Viewer** extension
(Florian Klampfer) — just click the file in the explorer.

Or query from the terminal:
```bash
# Quick query
sqlite3 ~/digitize_medical_records/VariableExtraction/outputs/research_database.db \
  "SELECT ID_Patient, Sexe, Age, gravite_palu, hemoglobine_J0 FROM patients;"

# All columns for one patient
sqlite3 ~/digitize_medical_records/VariableExtraction/outputs/research_database.db \
  "SELECT * FROM patients WHERE ID_Patient = 'PATIENT_002';"

# Count patients in database
sqlite3 ~/digitize_medical_records/VariableExtraction/outputs/research_database.db \
  "SELECT COUNT(*) FROM patients;"

# All severe malaria cases
sqlite3 ~/digitize_medical_records/VariableExtraction/outputs/research_database.db \
  "SELECT ID_Patient, Age, hemoglobine_J0, Parasitemie_J0
   FROM patients WHERE gravite_palu = 'grave';"
```

---

## Privacy & Data Security

| Component | Network access | Data sent externally |
|---|---|---|
| GLM-OCR | Local only | Never |
| Qwen 7B (classification + NER) | Local Ollama | Never |
| Qwen 72B (CNR extraction) | Local Ollama | Never |
| SQLite database | Local file | Never |
| Excel file | Local file | Never |
| Excel Viewer (VSCode) | None | Never |
| SQLite Viewer (VSCode) | None | Never |

**All processing is 100% local. No patient data ever leaves the server.**

---

## Adapting for a New Institution

Only edit **one file**: `VariableExtraction/variable_extraction_config.yaml`

Change only these 4 sections:
```yaml
institution:
  services: ["YOUR WARD NAMES"]          # e.g. "Pediatrie", "Urgences"

lieu_sejour_keywords:                     # travel destination keywords
  "country name": "Standardized Name"

chimio_keywords:                          # prophylaxis drug names
  "drug name": "Standardized Name"

treatment_keywords:                       # treatment drug names
  "drug name": "Standardized Name"
```

Everything else — OCR, classification, anonymization, J0/J3/J30 logic,
Excel formatting, SQLite schema — works without changes.

---

## Troubleshooting

**A patient is missing from the database (e.g. patient_001 not in SQLite):**
```bash
# Re-run variable extraction for ALL existing anonymized files:
python ~/digitize_medical_records/VariableExtraction/extract_variables.py \
  --all \
  --config ~/digitize_medical_records/VariableExtraction/variable_extraction_config.yaml \
  --output-dir ~/digitize_medical_records/VariableExtraction/outputs
```

**Pipeline fails on a patient:**
```bash
# Check what was extracted
cat outputs/patients/RDB_XXXX/RDB_XXXX_patient_raw.txt | head -50
```

**Variable extraction misses a value:**
Add or adjust patterns in `variable_extraction_config.yaml` under `fields:`.
No Python code changes needed. Then re-run variable extraction with --all (command above).

**Re-run only anonymization for one patient:**
```bash
python Anonymization_systemB/anonymize_patients.py --patient RDB_0186 --id 002
```