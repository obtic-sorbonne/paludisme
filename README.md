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

```markdown
# System B – CNR Paludisme Medical Digitization Pipeline

Fully automated pipeline: scanned PDFs → anonymized text → structured research table (Excel + SQLite).

**All processing is 100% local. No patient data ever leaves your server.**

---

## AI Models Required

This pipeline uses 5 AI models, all running locally via Ollama.

| Model | Size | Purpose |
|---|---|---|
| `glm-ocr` | 2.2 GB | OCR — extracts text from scanned PDF pages |
| `qwen2.5vl:7b` | 6.0 GB | Page classification (CNR vs non-CNR) + anonymization NER |
| `qwen2.5vl:72b` | 48 GB | Base model — required to build qwen72b-limited |
| `qwen72b-limited` | 48 GB | CNR form extraction (custom 22 GB VRAM version) |
| `qwen3:30b` | 18 GB | Variable extraction (Step 6) |

> **Why qwen72b-limited?**
> The full qwen2.5vl:72b uses 37 GB of VRAM. qwen72b-limited is a custom Ollama model built from it with `num_gpu 20`, reducing VRAM to 22 GB with only −1.6 percentage points accuracy loss. This frees ~15 GB for other researchers on shared GPU servers.

---
```

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
pip install pyyaml openpyxl pillow requests ollama
```

### Step 3 — Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 4 — Start Ollama (standard server)

If you have your own machine or dedicated GPU:

```bash
ollama serve &
sleep 5
```

### Step 4 (alternative) — Shared GPU server setup

If you share a GPU server with other researchers, run a personal Ollama instance on a different port to avoid conflicts:

```bash
# Create your own models directory (avoids permission issues with /var/snap/ollama)
mkdir -p ~/ollama_models

# Start personal Ollama on port 11435
CUDA_VISIBLE_DEVICES=0 \
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS=/home/YOUR_USERNAME/ollama_models \
ollama serve > /tmp/ollama_personal.log 2>&1 &
sleep 8

# Tell the pipeline to use your personal instance — add to ~/.bashrc
echo 'export OLLAMA_HOST=http://127.0.0.1:11435' >> ~/.bashrc
source ~/.bashrc
```

Replace `YOUR_USERNAME` with your Linux username (e.g. lfarooq).

### Step 5 — Download AI models (automatic)

The pipeline downloads all required models automatically on first run via `setup_models.sh`. You can also run it manually:

```bash
bash ~/digitize_medical_records/SystemB/setup_models.sh
```

This script checks each model, downloads missing ones, and automatically creates `qwen72b-limited` from `qwen2.5vl:72b`.

Or download manually:

```bash
# Fast models (~30 min total)
ollama pull glm-ocr
ollama pull qwen2.5vl:7b
ollama pull qwen3:30b

# Large model — takes several hours (48 GB)
ollama pull qwen2.5vl:72b

# Create the reduced VRAM version (fast — reuses existing weights)
cat > ~/Modelfile_limited << 'EOF'
FROM qwen2.5vl:72b
PARAMETER num_ctx 8192
PARAMETER num_gpu 20
EOF
ollama create qwen72b-limited -f ~/Modelfile_limited
```

### Step 6 — Verify all models

```bash
ollama list
# Should show: glm-ocr, qwen2.5vl:7b, qwen2.5vl:72b, qwen72b-limited, qwen3:30b
```

---

## Quick Start

```bash
source ~/digitize_medical_records/labelimg_env/bin/activate
bash ~/digitize_medical_records/SystemB/run_pipeline.sh "/path/to/patient_folder/"
```

| What you pass | What happens |
|---|---|
| Path to a single `.pdf` file | Processes that one PDF |
| Path to one patient folder | Processes all PDFs in that folder |
| Path to a folder of many patient folders | Processes every patient subfolder |

---

## Usage Examples

```bash
# Process one patient folder
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/path/to/patient_data/2006 RDB 0186/"

# Process ALL patients in a folder
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/path/to/patient_data/"

# Process 800 patient folders in one command
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/path/to/folder_with_800_patients/"

# Re-run variable extraction only (after editing config)
python ~/digitize_medical_records/SystemB/VariableExtraction/extract_variables_qwen.py \
  --all --host http://127.0.0.1:11434 --model qwen3:30b

# Skip variable extraction (OCR + anonymize only)
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/path/to/patients/" --skip-variables

# Skip OCR/extraction (variable extraction only)
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/path/to/patients/" --skip-extraction
```

---

## What Happens Step by Step

```
INPUT: Patient folder (e.g. "2006 RDB 0186/")
  Contains: DOC_00118.pdf, DOC_00119.pdf, DOC_00122.pdf ...

STEP 1: GLM-OCR  [model: glm-ocr]
  Each PDF page → raw text + page image.
  Already-processed pages are skipped automatically (cached).
  Cache uses patient_id + doc_id to prevent collisions when
  multiple patients have PDFs with identical filenames.

STEP 2: Page Classification  [model: qwen2.5vl:7b]
  Each page image → CNR Paludisme form OR clinical document?
  CNR forms have radio buttons and checkboxes.
  Clinical documents are lab results, letters, hospital reports.

STEP 3: Extraction
  CNR pages    → qwen72b-limited extracts all form fields visually
                 (species, dates, treatment, lab values, checkboxes)
                 Then postprocess_checker.py validates with GLM-OCR text.
  Non-CNR pages → GLM-OCR text only
                  (lab results, clinical reports, urgences letters)

STEP 4: Merge
  All document outputs merged into one file per patient.
  → outputs/patients/RDB_0186/RDB_0186_patient_raw.txt

STEP 5: Anonymization  [model: qwen2.5vl:7b for NER]
  Patient name  → [PATIENT_001]
  Doctor names  → [STAFF_001], [STAFF_002] ...
  Phone, email, address, NPI, FINESS → [ANONYMIZED]
  → outputs/final_anonymized/patient_001_anonymized.txt

STEP 6: Variable Extraction  [model: qwen3:30b]
  Runs ONCE at the end for ALL patients.
  Assigns pages to clinical timepoints (J0, J3, J30).
  Extracts 83 clinical variables per patient.
  → SystemB/VariableExtraction/outputs/research_table_qwen.xlsx
  → SystemB/VariableExtraction/outputs/research_database_qwen.db
```

---

## J3 Timepoint Selection Rules

Validated by CNR Paludisme scientists (May 2026):

**Rule 1 — Explicit label (highest priority):**
If a document page is explicitly labeled "Controle J3", "Suivi J3", or similar → that date is always used as J3, even if it falls within the J0 merge window. OCR typos like "Contole J3" (missing r) are handled automatically.

**Rule 2 — Data density fallback:**
If no explicit J3 label exists, the pipeline picks the date with the most complete laboratory results within the J0+2 to J0+5 day window. A "J4" with full data is preferred over a "J3" with empty results.

---

## Age Recording Rules

Validated by CNR Paludisme scientists (May 2026):

Always stored as decimal years:
- `10 mois` → `0.83` (10 ÷ 12)
- `22 mois` → `1.83`
- `5 ans et ½` → `5.5`
- `9 ans` → `9`

---

## Preparing Patient Documents

One folder per patient, containing all their PDFs:

```
my_patient_data/
├── 2006 RDB 0186/
│   ├── DOC_00118.pdf
│   ├── DOC_00119.pdf
│   └── DOC_00122.pdf
├── 2006 RDB 0201/
│   ├── DOC_00185.pdf
│   └── DOC_00192.pdf
```

- Folder names can be anything — used as patient identifiers
- Data can be located anywhere on your server
- Multiple patients can have PDFs with identical filenames — handled correctly via `patient_id + doc_id` cache keys

---

## Output Files

```
digitize_medical_records/
├── outputs/
│   ├── patient_registry.json          ← permanent folder → ID mapping
│   ├── patients/
│   │   └── RDB_0186/
│   │       ├── RDB_0186_patient_raw.txt
│   │       ├── RDB_0186_patient_anonymized.txt
│   │       └── RDB_0186_replacements.csv
│   └── final_anonymized/
│       ├── patient_001_anonymized.txt
│       └── ...
│
└── SystemB/
    └── VariableExtraction/
        └── outputs/
            ├── research_table_qwen.xlsx   ← one row per patient
            └── research_database_qwen.db  ← SQLite database
```

Every new patient is appended to the same Excel and SQLite. Re-running the same patient updates their row — no duplicates.

---

## Viewing Results

### Excel

```bash
# Download to your local machine:
scp username@server:~/digitize_medical_records/SystemB/VariableExtraction/outputs/research_table_qwen.xlsx ~/Desktop/
```

Or view in VSCode with the **Excel Viewer** extension (MESCIUS/GrapeCity).

### SQLite

```bash
sqlite3 ~/digitize_medical_records/SystemB/VariableExtraction/outputs/research_database_qwen.db \
  "SELECT ID_Patient, Sexe, Age, gravite_palu, hemoglobine_J0 FROM patients;"
```

Or view in VSCode with the **SQLite Viewer** extension (Florian Klampfer).

---

## Patient ID System

Each patient folder gets a permanent sequential ID on first processing, saved in `outputs/patient_registry.json`:

```json
{
  "RDB_0186": "001",
  "RDB_0201": "002",
  "RDB_0015": "003"
}
```

Re-running the same folder always reuses the same ID. No duplicates.

---

## Privacy & Data Security

| Component | Network access | Data sent externally |
|---|---|---|
| GLM-OCR | Local Ollama only | Never |
| qwen2.5vl:7b (classify + anonymize) | Local Ollama only | Never |
| qwen72b-limited (CNR extraction) | Local Ollama only | Never |
| qwen3:30b (variable extraction) | Local Ollama only | Never |
| Excel / SQLite output | Local file | Never |

**No patient data ever leaves your server.**

---

## Adapting for a New Institution

Edit only one file: `SystemB/VariableExtraction/variable_extraction_config.yaml`

```yaml
institution:
  services:
    - "YOUR WARD NAME"       # e.g. "Pediatrie", "Urgences"

lieu_sejour_keywords:
  "country keyword": "Standardized Country Name"

chimio_keywords:
  "drug keyword": "Standardized Drug Name"

treatment_keywords:
  "drug keyword": "Standardized Drug Name"
```

No Python code changes needed. Re-run variable extraction with `--all` after editing.

> **Note on French accents:** OCR sometimes drops accents (e.g. `Symptômes` → `Symptomes`). The pipeline handles this automatically using accent-normalized matching.

---

## Known Limitations

- `fièvre_J0` reflects whether fever was documented in clinical notes, not necessarily the chief complaint
- J3 date follows the two-rule system above (explicit label → data density)
- Variables left blank when not documented in the scanned records
- On shared GPU servers with limited free VRAM, page classification and CNR extraction may fail — ensure enough GPU memory is available before running
- Anonymization NER (doctor name detection in free text) requires qwen2.5vl:7b to load; if GPU is full, patient name is still anonymized via regex but some doctor names in narrative text may not be caught

---

## Troubleshooting

**Models not downloading (permission denied):**
```bash
mkdir -p ~/ollama_models
OLLAMA_MODELS=~/ollama_models ollama pull glm-ocr
```

**qwen72b-limited missing after server restart:**
```bash
cat > ~/Modelfile_limited << 'EOF'
FROM qwen2.5vl:72b
PARAMETER num_ctx 8192
PARAMETER num_gpu 20
EOF
ollama create qwen72b-limited -f ~/Modelfile_limited
```

**Page classification fails (model failed to load):**
GPU memory is full. Check with `nvidia-smi`. Free memory or wait for other processes to finish before running the pipeline.

**Patient missing from database:**
```bash
python ~/digitize_medical_records/SystemB/VariableExtraction/extract_variables_qwen.py \
  --all --host http://127.0.0.1:11434 --model qwen3:30b
```

**Ollama not running:**
```bash
ollama serve &
sleep 5
ollama list
```

**Shared server — pipeline using wrong Ollama port:**
```bash
export OLLAMA_HOST=http://127.0.0.1:11435
echo 'export OLLAMA_HOST=http://127.0.0.1:11435' >> ~/.bashrc
```

**Check patient ID mapping:**
```bash
cat ~/digitize_medical_records/outputs/patient_registry.json
```

---

## Repository Structure

```
paludisme/
├── SystemA/                           ← First generation pipeline (PaddleOCR)
├── SystemB/                           ← This pipeline
│   ├── README.md                      ← This file
│   ├── run_pipeline.sh                ← MAIN ENTRY POINT
│   ├── setup_models.sh                ← Auto-download all 5 AI models
│   ├── Anonymization_systemB/         ← Anonymization module
│   ├── CNR_forms/                     ← GLM-OCR + CNR form extraction
│   ├── SystemB_page_classification/   ← Page classifier + orchestrator
│   └── VariableExtraction/
│       ├── extract_variables_qwen.py  ← Main extraction script (qwen3:30b)
│       ├── extraction_fields.yaml     ← Q&A field definitions (edit to add variables)
│       └── variable_extraction_config.yaml ← Institution config
├── .gitignore
└── requirements.txt
```

---

## Contact & Support

Developed by **Labiba FAROOQ** at **ObTIC, Sorbonne Université**.
Supervisor: **Motasem Alrahabi**
Internship: March–July 2026, Paris-Robert Debré Hospital / CNR Paludisme

GitHub: https://github.com/obtic-sorbonne/paludisme


