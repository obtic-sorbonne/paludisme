# Anonymisation de documents médicaux

Pseudonymisation automatique de dossiers médicaux PDF scannés (français). Extrait le texte par OCR, identifie le patient, remplace son nom par un identifiant (`[PATIENT_001]`) et supprime toutes les autres données personnelles (`[ANONYMIZED]`).

## Installation

```bash
# Créer et activer un environnement virtuel
python3 -m venv .medrec
source .medrec/bin/activate

# PyTorch (adapter cu124 selon votre version CUDA — vérifier avec nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Dépendances
pip install docling easyocr spacy
python -m spacy download fr_core_news_lg

# Pré-télécharger les modèles Docling pour usage hors-ligne (confidentialité des données)
docling-tools models download
```

## Structure des données en entrée

```
data/
├── patient_001/          # un sous-dossier par patient
│   ├── DOC_001.pdf
│   ├── DOC_002.pdf
│   └── DOC_003.pdf
├── patient_002/
│   └── DOC_004.pdf
```

## Utilisation

```bash
# Pipeline complet (avec GPU)
python -m anonymization -i data/ -o output/ --gpu

# Pipeline complet (CPU uniquement)
python -m anonymization -i data/ -o output/

# Tester l'OCR seul sur un dossier (avec GPU)
python anonymization/ocr.py data/patient_001/ output_ocr/ --gpu
```

## Résultats produits

```
output/
├── patient_001/
│   ├── all_documents.txt   # texte anonymisé (tous les PDFs concaténés)
│   └── replacements.csv    # détail de chaque remplacement effectué
├── patient_002/
│   └── ...
├── mapping.csv             # correspondance patient_id ↔ nom réel
└── anonymization_report.txt
```

**mapping.csv** — table de correspondance (à conserver séparément, contient les vrais noms) :

| subfolder | patient_id | real_lastnames | real_firstnames | num_docs |
|---|---|---|---|---|
| patient_001 | PATIENT_001 | DUPONT | JEAN | 3 |

**replacements.csv** — détail par fichier (pour vérifier les faux positifs). SpaCy utilisé pour NER se trompe en mélangeant les noms des personnes et des médicaments.

| file | category | original | replacement |
|---|---|---|---|
| DOC_001.pdf | patient_name | DUPONT | [PATIENT_001] |
| DOC_001.pdf | TELEPHONE | 01.40.03.20.48 | [ANONYMIZED] |
| DOC_001.pdf | other_person | Dr Albert FAYE | [ANONYMIZED] |

## Ce qui est anonymisé

- **Nom du patient** → `[PATIENT_NNN]`
- **Autres personnes** (médecins, internes, famille) → `[ANONYMIZED]`
- **Téléphones, emails, adresses, identifiants (NPI, IP, NIR)** → `[ANONYMIZED]`

## Ce qui est préservé

- Date de naissance, âge, sexe
- Noms d'hôpitaux et codes FINESS
- Dates médicales (consultations, hospitalisations)
- Médicaments et termes médicaux

## Tests

```bash
python test_pipeline.py
```

## Architecture

```
anonymization/
├── ocr.py               # Docling + EasyOCR (français, GPU)
├── name_extraction.py   # extraction du nom patient depuis les en-têtes
├── pii.py               # détection PII (regex) + noms d'autres personnes
├── pseudonyms.py        # génération [PATIENT_NNN]
├── anonymizer.py        # moteur d'anonymisation (4 couches)
└── pipeline.py          # orchestrateur principal
```