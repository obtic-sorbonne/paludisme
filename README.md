# Paludisme – Medical Document Digitization Pipeline

Automated pipeline for digitizing CNR Paludisme malaria survey records.

## Systems

- **SystemA/** — First generation pipeline (PaddleOCR based)
- **SystemB/** — Second generation pipeline (GLM-OCR + Qwen AI, fully automated)

## Getting Started

Developed at **ObTIC, Sorbonne Université** by **Labiba FAROOQ**.


# System B – CNR Paludisme Medical Digitization Pipeline

Fully automated pipeline: scanned PDFs → anonymized text → structured research table (Excel + SQLite).

**All processing is 100% local. No patient data ever leaves your server.**

---

## Installation

### Step 1 — Clone the repository

```bash
# Important: clone into a folder named exactly "digitize_medical_records"
git clone https://github.com/obtic-sorbonne/paludisme.git digitize_medical_records
cd digitize_medical_records
```

### Step 2 — Create Python environment

```bash
python3 -m venv labelimg_env
source labelimg_env/bin/activate
pip install pyyaml openpyxl pillow requests
```

### Step 3 — Install Ollama and download AI models

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve &
sleep 5

# Download the 3 required models
ollama pull glm-ocr           # OCR model (~4 GB)
ollama pull qwen2.5vl:7b      # Page classification + anonymization (~6 GB)
ollama pull qwen72b-limited   # CNR form extraction (~48 GB)
```

> **Note:** `qwen72b-limited` may need to be loaded from a local file on your server.
> Ask your system administrator if it is not available via `ollama pull`.

### Step 4 — Verify setup

```bash
ollama list
# Should show all 3 models

python --version
# Should show Python 3.10 or higher
```

---

## Quick Start

```bash
# One command does everything — auto-detects what you give it:
bash ~/digitize_medical_records/SystemB/run_pipeline.sh "<path>"
```

| What you pass | What happens |
|---|---|
| Path to a single `.pdf` file | Processes that one PDF |
| Path to one patient folder | Processes all PDFs in that folder |
| Path to a folder containing many patient folders | Processes every patient subfolder |

---

## Usage Examples

```bash
# ── Process one patient folder ──────────────────────────────────────────────
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/path/to/your/patient_data/2006 RDB 0186/"

# ── Process ALL patients in a folder ────────────────────────────────────────
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/path/to/your/patient_data/"

# ── Process 800 patient folders in one command ───────────────────────────────
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/path/to/folder_with_800_patient_folders/"
# This loops through all 800 subfolders, processes each one
# (OCR → classify → extract → anonymize), then writes all 800 rows
# into the same Excel and SQLite. One command for everything.

# ── Re-run a patient already processed (safe — keeps same patient ID) ────────
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/path/to/your/patient_data/2006 RDB 0186/"

# ── Re-run ONLY variable extraction (e.g. after editing the config file) ─────
python ~/digitize_medical_records/SystemB/VariableExtraction/extract_variables.py \
  --all \
  --config ~/digitize_medical_records/SystemB/VariableExtraction/variable_extraction_config.yaml \
  --output-dir ~/digitize_medical_records/SystemB/VariableExtraction/outputs

# ── Run pipeline but skip Excel/SQLite output ─────────────────────────────────
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/path/to/patients/" --skip-variables
```

---

## Preparing Your Patient Documents

Organise your PDF files with **one folder per patient**:

```
my_patient_data/
├── 2006 RDB 0186/
│   ├── DOC_00118.pdf
│   ├── DOC_00119.pdf
│   └── DOC_00122.pdf
├── 2006 RDB 0201/
│   ├── DOC_00185.pdf
│   └── DOC_00192.pdf
└── 2006 RDB 0189/
    └── DOC_00177.pdf
```

The folder names can be anything — the system uses them as patient identifiers.
The data can be located **anywhere on your server** — it does not need to be inside the `digitize_medical_records` folder.

---

## What Happens Step by Step

```
INPUT: Patient folder (e.g. "2006 RDB 0186/")
  └── Contains: DOC_00118.pdf, DOC_00119.pdf, DOC_00122.pdf ...

STEP 1: GLM-OCR
  Each PDF is read page by page using GLM-OCR.
  Produces: raw text + page images.
  Already-processed documents are skipped automatically.

STEP 2: Page Classification (Qwen 7B)
  Each page image is shown to Qwen 7B which decides:
  "Is this a CNR Paludisme form or a clinical document?"

STEP 3: Extraction
  ├── CNR pages    → Qwen 72B extracts all form fields
  │                  (species, dates, treatment, lab values, etc.)
  │                  Then postprocess_checker.py validates the output.
  └── Non-CNR pages → GLM-OCR text only
                      (lab results, clinical reports, urgences)

STEP 4: Merge
  All document outputs merged into one file per patient.
  Output: outputs/patients/RDB_0186/RDB_0186_patient_raw.txt

STEP 5: Anonymization
  All personal identifiers are removed:
    - Patient name  → [PATIENT_001]
    - Doctor names  → [STAFF_001], [STAFF_002] ...
    - Phone, email, address, NPI, FINESS → [ANONYMIZED]
  Uses Qwen 7B NER (local, no internet) to detect names in free text.
  Each patient folder gets a permanent sequential ID stored in
  outputs/patient_registry.json — re-running the same folder always
  reuses the same ID, never creates duplicates.
  Outputs:
    outputs/patients/RDB_0186/RDB_0186_patient_anonymized.txt
    outputs/patients/RDB_0186/RDB_0186_replacements.csv
    outputs/final_anonymized/patient_001_anonymized.txt

STEP 6: Variable Extraction
  Runs ONCE at the end for ALL patients together.
  Assigns each document/page to J0, J3, or J30:
    - J0  = first consultation date (admission)
    - J3  = first follow-up date more than 3 days from J0
    - J30 = last consultation date
    - Dates within 3 days of J0 are merged into J0
  Extracts 130 clinical variables per patient.
  Writes one row per patient to:
    SystemB/VariableExtraction/outputs/research_table.xlsx
    SystemB/VariableExtraction/outputs/research_database.db
```

---

## Output Files

```
digitize_medical_records/
├── outputs/
│   ├── patient_registry.json             ← permanent folder → ID mapping
│   ├── patients/
│   │   ├── RDB_0186/
│   │   │   ├── RDB_0186_patient_raw.txt
│   │   │   ├── RDB_0186_patient_anonymized.txt
│   │   │   └── RDB_0186_replacements.csv
│   │   └── RDB_XXXX/ ...
│   └── final_anonymized/
│       ├── patient_001_anonymized.txt
│       ├── patient_002_anonymized.txt
│       └── ...
│
└── SystemB/
    └── VariableExtraction/
        └── outputs/
            ├── research_table.xlsx    ← ONE file, all patients (one row each)
            └── research_database.db  ← ONE database, all patients
```

> **Important:** Every new patient is **appended** to the same Excel file and
> the same SQLite database. Running on 15 patients gives 15 rows in one sheet.
> Running again on the same patient **updates** their row — no duplicates.

---

## Viewing Results

### Excel
Download to your local machine:
```bash
# Run this on your LOCAL machine terminal:
scp username@your-server:~/digitize_medical_records/SystemB/VariableExtraction/outputs/research_table.xlsx ~/Desktop/
```
Then open in Excel or LibreOffice.

Or view directly in VSCode using the **Excel Viewer** extension (MESCIUS/GrapeCity).

### SQLite
View in VSCode using the **SQLite Viewer** extension (Florian Klampfer) — just click the `.db` file.

Or query from the terminal:
```bash
sqlite3 ~/digitize_medical_records/SystemB/VariableExtraction/outputs/research_database.db \
  "SELECT ID_Patient, Sexe, Age, gravite_palu, hemoglobine_J0 FROM patients;"

sqlite3 ~/digitize_medical_records/SystemB/VariableExtraction/outputs/research_database.db \
  "SELECT COUNT(*) FROM patients;"
```

---

## Patient ID System

Each patient folder is assigned a **permanent sequential ID** the first time it is processed.
This mapping is saved in `outputs/patient_registry.json`:

```json
{
  "RDB_0186": "001",
  "RDB_0201": "002",
  "RDB_0189": "003"
}
```

- Re-running the same folder always reuses the same ID
- New folders get the next sequential number automatically
- No duplicates, no manual management needed

---

## Privacy & Data Security

| Component | Network access | Data sent externally |
|---|---|---|
| GLM-OCR | Local Ollama only | Never |
| Qwen 7B (classification + anonymization) | Local Ollama only | Never |
| Qwen 72B (CNR form extraction) | Local Ollama only | Never |
| Excel output | Local file | Never |
| SQLite database | Local file | Never |

**No patient data ever leaves your server at any point.**

---

## Adapting for a New Institution

Only edit **one file**: `SystemB/VariableExtraction/variable_extraction_config.yaml`

Change only these 4 sections at the top of the file:

```yaml
institution:
  services:
    - "YOUR WARD NAME"       # e.g. "Pediatrie", "Urgences"

lieu_sejour_keywords:         # travel destination keywords → standardized names
  "country keyword": "Standardized Country Name"

chimio_keywords:              # prophylaxis drug names
  "drug keyword": "Standardized Drug Name"

treatment_keywords:           # treatment drug names
  "drug keyword": "Standardized Drug Name"
```

Everything else — OCR, classification, anonymization, J0/J3/J30 logic,
Excel formatting, SQLite schema — works without any changes.

> **Note on French accents:** OCR sometimes drops accents (e.g. `Symptômes` → `Symptomes`,
> `Consuliation` instead of `Consultation`). The pipeline handles this automatically
> using accent-normalized matching. No manual correction needed.

---

## Known Limitations

- `fièvre_J0` reflects whether fever was documented in the clinical notes, not necessarily the chief complaint
- J3 date = first follow-up date more than 3 days from admission (may vary ±1 day depending on document dates)
- Variables left blank when not documented in the scanned records
- CNR form fields take lower priority than clinical document values (configurable per field)

---

## Troubleshooting

**A patient is missing from the database:**
```bash
python ~/digitize_medical_records/SystemB/VariableExtraction/extract_variables.py \
  --all \
  --config ~/digitize_medical_records/SystemB/VariableExtraction/variable_extraction_config.yaml \
  --output-dir ~/digitize_medical_records/SystemB/VariableExtraction/outputs
```

**Check which folder maps to which patient ID:**
```bash
cat ~/digitize_medical_records/outputs/patient_registry.json
```

**Pipeline fails on a patient — check what was extracted:**
```bash
cat ~/digitize_medical_records/outputs/patients/RDB_XXXX/RDB_XXXX_patient_raw.txt | head -50
```

**Variable extraction misses a value:**
Edit `SystemB/VariableExtraction/variable_extraction_config.yaml` under `fields:`.
No Python code changes needed. Then re-run variable extraction with `--all`.

**Ollama not running:**
```bash
ollama serve &
sleep 3
ollama list
```

---

## Repository Structure

```
paludisme/
├── SystemA/                          ← System A (previous pipeline)
├── SystemB/                          ← System B (this pipeline)
│   ├── run_pipeline.sh               ← MAIN ENTRY POINT
│   ├── Anonymization_systemB/        ← anonymization module
│   ├── CNR_forms/                    ← OCR + CNR extraction scripts
│   ├── SystemB_page_classification/  ← page classifier + orchestrator
│   └── VariableExtraction/           ← 130-variable extractor + config
├── .gitignore
└── requirements.txt
```

---

## Contact & Support

Developed by **Labiba FAROOQ** at **ObTIC, Sorbonne Université**.

GitHub: https://github.com/obtic-sorbonne/paludisme
