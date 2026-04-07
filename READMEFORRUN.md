
# Page classification / routing pipeline

This pipeline is the first-stage routing layer for a PDF. It:

* creates per-page Paddle JSON files
* classifies each page type
* creates routing decisions for each page

It does **not yet** run the downstream extraction pipelines automatically.

## Files used

### Wrapper scripts

* `page_classification_pipeline/run_page_classification_pipeline.sh`
* `page_classification_pipeline/batch_run_page_classification_pipeline.sh`

### Core scripts

* `page_classification_pipeline/table_paddle_test.py`
* `page_classification_pipeline/classify_page_type.py`
* `page_classification_pipeline/route_page_processing.py`

## One PDF

```bash
cd ~/digitize_medical_records
./page_classification_pipeline/run_page_classification_pipeline.sh "/absolute/path/to/file.pdf"
```

### Example

```bash
cd ~/digitize_medical_records
./page_classification_pipeline/run_page_classification_pipeline.sh "/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0185/DOC_00116.pdf"
```

## Many PDFs from one folder

```bash
cd ~/digitize_medical_records
./page_classification_pipeline/batch_run_page_classification_pipeline.sh "/absolute/path/to/folder"
```

### Example

```bash
cd ~/digitize_medical_records
./page_classification_pipeline/batch_run_page_classification_pipeline.sh "/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0185"
```

## Internal steps

1. Run `table_paddle_test.py` to create per-page Paddle JSON files
2. Run `classify_page_type.py` on each page JSON
3. Run `route_page_processing.py` using the page JSON and classification JSON

## Next branch decision

* `clinical_report_page` → narrative pipeline
* `lab_table_page` → lab table pipeline
* `form_page` → form field extraction
* `unknown` → keep OCR text and review

## Main outputs

```text
/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle_table/<DOC_STEM>_*_res.json
/home/lfarooq/digitize_medical_records/benchmark_outputs/page_classification/<DOC_STEM>_*_page_type.json
/home/lfarooq/digitize_medical_records/benchmark_outputs/page_routing/<DOC_STEM>_*_route.json
```









Narrative report FORM :
running command:
cd ~/digitize_medical_records
source env_paddle/bin/activate

PDF_DIR="/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0156"

for pdf in "$PDF_DIR"/*.pdf; do
  echo "=============================="
  echo "Processing PDF: $pdf"
  echo "=============================="

  python table_paddle_test.py "$pdf"

  doc_stem=$(basename "$pdf" .pdf)

  for json in /home/lfarooq/digitize_medical_records/benchmark_outputs/paddle_table/${doc_stem}_*_res.json; do
    [ -e "$json" ] || continue

    python classify_page_type.py "$json"

    class_json="/home/lfarooq/digitize_medical_records/benchmark_outputs/page_classification/$(basename "$json" .json)_page_type.json"

    python route_page_processing.py "$json" "$class_json"
  done

  python narrative_clinical_pipeline/ocr_paddle_test.py "$pdf"
done

-----main running line:   python narrative_clinical_pipeline/ocr_paddle_test.py "$pdf"







# Narrative Clinical Pipeline with tables

This pipeline runs PaddleOCR on one narrative clinical PDF and writes the extracted text plus reconstructed tables to:

`/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle/<DOC_STEM>.txt`

Use the single-file script for one PDF, or the batch wrapper for a whole folder of PDFs.


----How to run the wrapper (whole file with 1 command)

### One PDF

```bash
cd ~/digitize_medical_records
./narrative_clinical_pipeline/run_narrative_layout_pipeline.sh "/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0186/DOC_00119.pdf"
```

### Many PDFs from one folder

```bash
cd ~/digitize_medical_records
./narrative_clinical_pipeline/batch_run_narrative_layout_pipeline.sh "/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0186"
```





















Lab tables FORM pipeline from raw PDF

1. Run PP-Structure on the PDF
   Script: pp_structure_test.py

2. Run Paddle table recognition on the PDF
   Script: table_paddle_test.py

3. Run TATR table detection on the PDF
   Script: tatr_detect_table.py

4. Merge PP-Structure and Paddle outputs
   Script: merge_ppstructure_with_paddle_ocr.py

5. Build hybrid table output using PP-Structure + Paddle + TATR
   Script: hybrid_table_parser.py

6. Clean hybrid output into final hybrid txt
   Script: final_hybrid_table_cleaner.py

7. Convert final hybrid txt into CSV / HTML / XLSX
   Script: improve_final_table_output.py


----running command:

cd ~/digitize_medical_records
source env_paddle/bin/activate

python pp_structure_test.py \
"/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0156/DOC_00096.pdf"

python table_paddle_test.py \
"/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0156/DOC_00096.pdf"

python tatr_detect_table.py \
"/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0156/DOC_00096.pdf" \
--page-index 0

python merge_ppstructure_with_paddle_ocr.py \
"/home/lfarooq/digitize_medical_records/benchmark_outputs/pp_structure/DOC_00096_0_res.json" \
"/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle_table/DOC_00096_0_0_res.json"

python hybrid_table_parser.py \
"/home/lfarooq/digitize_medical_records/benchmark_outputs/pp_structure/DOC_00096_0_res.json" \
"/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle_table/DOC_00096_0_0_res.json" \
"/home/lfarooq/digitize_medical_records/benchmark_outputs/tatr/DOC_00096_p1_detections.txt"

python final_hybrid_table_cleaner.py \
"/home/lfarooq/digitize_medical_records/benchmark_outputs/hybrid_tables/DOC_00096_0_res_hybrid.txt"

python improve_final_table_output.py \
"/home/lfarooq/digitize_medical_records/benchmark_outputs/hybrid_tables_final/DOC_00096_0_res_hybrid_final.txt"


----some info:

pp_structure_test.py
Runs PP-Structure on the PDF to detect page layout and extract table structure/content. It gives a structured JSON view of the table area.
table_paddle_test.py
Runs Paddle table/OCR processing on the PDF to extract the table text and recognition results. It gives another OCR-based table output that complements PP-Structure.
tatr_detect_table.py
Uses Table Transformer (TATR) to detect the table region on a page. This helps the hybrid parser know where the table is and improves alignment.
merge_ppstructure_with_paddle_ocr.py
Combines PP-Structure output with Paddle OCR output into one merged table text file. This step tries to keep the best information from both sources.
hybrid_table_parser.py
Builds a hybrid table representation using PP-Structure, Paddle OCR, and TATR detections together. This is the main step that turns multiple raw outputs into a cleaner table.
final_hybrid_table_cleaner.py
Cleans and normalizes the hybrid table text, removing noise and organizing rows/notes more clearly. It produces the final intermediate text used for formatting.
improve_final_table_output.py
Converts the cleaned final text into user-friendly output files like CSV, HTML, and XLSX. It also formats notes and section titles for easier reading.
parse_paddle_table_results.py
Converts raw Paddle table JSON into a readable text form for inspection/debugging. It is mostly a helper script, not the main final-output step.  python narrative_clinical_pipeline/ocr_paddle_test.py "$pdf"


----How to run the wrapper (whole file with 1 command)

### One PDF

```bash
cd ~/digitize_medical_records
./lab_tables_pipeline/run_lab_table_pipeline.sh "/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0156/DOC_00096.pdf" 0
```

### Many PDFs from one folder

```bash
cd ~/digitize_medical_records
./lab_tables_pipeline/batch_run_lab_table_pipeline.sh "/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0156" 0
```





















# Checkbox / form full-document pipeline

This pipeline runs OCR and structured parsing for checkbox-based CNR malaria forms. It creates:

* per-page Paddle JSON files
* rendered page PNG files for visual fallback
* a full OCR TXT
* final structured outputs for the whole document

## Files used

### Main wrapper scripts

* `checkbox_full_doc_pipeline/run_checkbox_form_pipeline.sh`
* `checkbox_full_doc_pipeline/batch_run_checkbox_form_pipeline.sh`

### Main parser

* `checkbox_full_doc_pipeline/parse_full_document_all_pages.py`

### Page-specific OCR text parsers

* `checkbox_full_doc_pipeline/parse_page1_from_ocr_text.py`
* `checkbox_full_doc_pipeline/parse_page2_from_ocr_text.py`
* `checkbox_full_doc_pipeline/parse_page3_from_ocr_text.py`
* `checkbox_full_doc_pipeline/parse_page4_from_ocr_text.py`

### Shared utilities

* `checkbox_full_doc_pipeline/cnr_common.py`

### Visual fallback / helper parsers

* `checkbox_full_doc_pipeline/parse_radio_visual_from_ocr_json.py`
* `checkbox_full_doc_pipeline/debug_visual_utils.py`
* `checkbox_full_doc_pipeline/parse_page4_controle_parasito_visual.py`
* `checkbox_full_doc_pipeline/parse_option_groups_visual_generic.py`

### Upstream OCR / page-assets generation

* `table_paddle_test.py`
* `narrative_clinical_pipeline/ocr_paddle_test.py`

## One PDF

```bash id="t9h68p"
cd ~/digitize_medical_records
./checkbox_full_doc_pipeline/run_checkbox_form_pipeline.sh "/absolute/path/to/file.pdf"
```

### Example

```bash id="oi3w6g"
cd ~/digitize_medical_records
./checkbox_full_doc_pipeline/run_checkbox_form_pipeline.sh "/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0185/DOC_00116.pdf"
```

## Many PDFs from one folder

```bash id="urjlwm"
cd ~/digitize_medical_records
./checkbox_full_doc_pipeline/batch_run_checkbox_form_pipeline.sh "/absolute/path/to/folder"
```

### Example

```bash id="qkf0ta"
cd ~/digitize_medical_records
./checkbox_full_doc_pipeline/batch_run_checkbox_form_pipeline.sh "/home/lfarooq/digitize_medical_records/data/medrecords/2007_scan_output_fixed/Batch de docs corrigés - Copie/2006 RDB 0185"
```

## Internal steps run by the wrapper

1. Run `table_paddle_test.py` on the PDF to create per-page Paddle JSON files
2. Render PDF pages to PNG using `pdftoppm` for visual fallback parsers
3. Run `narrative_clinical_pipeline/ocr_paddle_test.py` to create the OCR TXT
4. Run `checkbox_full_doc_pipeline/parse_full_document_all_pages.py` on the OCR TXT

## Main outputs

For a PDF named `DOC_00116.pdf`, the main outputs are:

```text id="t5wiiu"
/home/lfarooq/digitize_medical_records/benchmark_outputs/full_document_all_pages_parser/DOC_00116_all_pages_all_specs.json
/home/lfarooq/digitize_medical_records/benchmark_outputs/full_document_all_pages_parser/DOC_00116_all_pages_all_specs_summary.txt
/home/lfarooq/digitize_medical_records/benchmark_outputs/full_document_all_pages_parser/DOC_00116_full_final_output.txt
```

## Supporting generated files

The wrapper also produces supporting files used internally by the parser:

```text id="3h0lks"
/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle_table/<DOC_STEM>_*_res.json
/home/lfarooq/digitize_medical_records/benchmark_outputs/form_visual_pages/<DOC_STEM>-*.png
/home/lfarooq/digitize_medical_records/benchmark_outputs/paddle/<DOC_STEM>.txt
```


















## Final Unified Document Pipeline

This project now includes a **final unified wrapper pipeline** that processes a full medical PDF document with a single command and produces one merged final text output.

The goal of this wrapper is to automatically:
1. run OCR/layout extraction page by page,
2. classify each page by document type,
3. route each page to the correct extraction branch,
4. merge all extracted outputs into one final document-level TXT and JSON.

---

## Overall pipeline logic

The final pipeline works in the following order:

### 1. Page OCR + layout extraction
The pipeline first generates Paddle-based OCR/layout JSON files for all pages of the PDF.

These files are stored in:

- `benchmark_outputs/paddle_table/`

Each page gets its own JSON output containing:
- OCR text
- layout blocks
- table detection metadata
- image/text/table flags

---

### 2. Page classification
Each page is then classified into one of the following main page types:

- `form_page`
- `clinical_report_page`
- `lab_table_page`
- `unknown`

The classifier uses:
- OCR text content
- layout signals
- page-specific keywords and structural cues

Classification outputs are stored in:

- `benchmark_outputs/page_classification/`

Routing outputs are stored in:

- `benchmark_outputs/page_routing/`

---

### 3. Page routing
After classification, each page is routed to the correct downstream extraction branch:

- `form_page` → form field extraction
- `clinical_report_page` → report text extraction
- `lab_table_page` → lab table extraction
- `unknown` → keep OCR text only

This means the wrapper does **not** treat every page the same way.  
Instead, it decides page by page which extractor should be used.

---

### 4. Extraction branches

#### A. Form pages
Form pages are processed through the checkbox/form extraction pipeline.

This is used for pages such as:
- CNR checkbox forms
- structured paludisme forms
- pages containing yes/no/NSP options
- pages with medical form fields

Output is preserved in text form and can also produce structured summaries.

---

#### B. Clinical report pages
Narrative/clinical pages are processed as text-heavy medical reports.

This branch is used for:
- emergency reports
- hospitalization reports
- consultation reports
- narrative pages containing embedded medical explanations

These pages are extracted mainly as OCR text, with optional table reformatting when useful.

---

#### C. Lab table pages
Lab report pages are routed to the lab table pipeline.

This branch is used for pages containing structured lab tables, such as:
- hematology tables
- biochemistry tables
- parasitology result tables

The lab branch currently works best for the standard hematology-style tables and is being extended to support additional table variants without breaking the current working format.

---

#### D. Unknown pages
If a page does not strongly match a known category, it is kept as OCR text only.

This avoids forcing a wrong parser on pages that would otherwise fail or produce damaged output.

---

### 5. Final merge step
After all pages are processed, the wrapper merges the page-level outputs into one final document-level result.

Final outputs are stored in:

- `benchmark_outputs/final_document_pipeline/<DOC_STEM>/merged_final_output.txt`
- `benchmark_outputs/final_document_pipeline/<DOC_STEM>/merged_final_output.json`
- `benchmark_outputs/final_document_pipeline/<DOC_STEM>/pipeline.log`

This gives one final combined output for the whole PDF.

---

## Main command to run the final pipeline

Run the final wrapper pipeline with:

```bash
cd ~/digitize_medical_records

./unified_document_pipeline/run_document_pipeline.sh \
"/absolute/path/to/file.pdf"