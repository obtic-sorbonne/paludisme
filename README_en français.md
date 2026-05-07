# Paludisme – Pipeline de Numérisation de Documents Médicaux

Pipeline automatisé pour la numérisation des dossiers d'enquête CNR Paludisme.

## Systèmes

- **SystemA/** — Pipeline de première génération (basé sur PaddleOCR)
- **SystemB/** — Pipeline de deuxième génération (GLM-OCR + Qwen IA, entièrement automatisé)

## Pour Commencer

Développé à **ObTIC, Sorbonne Université** par **Labiba FAROOQ**.


# Système B – Pipeline de Numérisation de Documents Médicaux CNR Paludisme

Pipeline entièrement automatisé : PDFs scannés → texte anonymisé → tableau de recherche structuré (Excel + SQLite).

**Tout le traitement est 100% local. Aucune donnée patient ne quitte jamais votre serveur.**

---

## Installation

### Étape 1 — Cloner le dépôt

```bash
# Important : cloner dans un dossier nommé exactement "digitize_medical_records"
git clone https://github.com/obtic-sorbonne/paludisme.git digitize_medical_records
cd digitize_medical_records
```

### Étape 2 — Créer l'environnement Python

```bash
python3 -m venv labelimg_env
source labelimg_env/bin/activate
pip install pyyaml openpyxl pillow requests
```

### Étape 3 — Installer Ollama et télécharger les modèles IA

```bash
# Installer Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Démarrer Ollama
ollama serve &
sleep 5

# Télécharger les 3 modèles requis
ollama pull glm-ocr           # Modèle OCR (~4 Go)
ollama pull qwen2.5vl:7b      # Classification des pages + anonymisation (~6 Go)
ollama pull qwen72b-limited   # Extraction des formulaires CNR (~48 Go)
```

> **Note :** `qwen72b-limited` doit peut-être être chargé depuis un fichier local sur votre serveur.
> Contactez votre administrateur système s'il n'est pas disponible via `ollama pull`.

### Étape 4 — Vérifier l'installation

```bash
ollama list
# Doit afficher les 3 modèles

python --version
# Doit afficher Python 3.10 ou supérieur
```

---

## Démarrage Rapide

```bash
# Une seule commande fait tout — détecte automatiquement ce que vous lui donnez :
bash ~/digitize_medical_records/SystemB/run_pipeline.sh "<chemin>"
```

| Ce que vous passez | Ce qui se passe |
|---|---|
| Chemin vers un fichier `.pdf` | Traite ce PDF uniquement |
| Chemin vers un dossier patient | Traite tous les PDFs dans ce dossier |
| Chemin vers un dossier contenant plusieurs dossiers patients | Traite chaque sous-dossier patient |

---

## Exemples d'Utilisation

```bash
# ── Traiter un seul dossier patient ─────────────────────────────────────────
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/chemin/vers/vos/donnees/2006 RDB 0186/"

# ── Traiter TOUS les patients dans un dossier ───────────────────────────────
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/chemin/vers/vos/donnees/"

# ── Traiter 800 dossiers patients en une seule commande ─────────────────────
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/chemin/vers/le/dossier_avec_800_patients/"
# Cette commande parcourt les 800 sous-dossiers, traite chacun
# (OCR → classification → extraction → anonymisation), puis écrit
# les 800 lignes dans le même Excel et SQLite. Une seule commande suffit.

# ── Retraiter un patient déjà traité (sûr — conserve le même ID) ────────────
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/chemin/vers/vos/donnees/2006 RDB 0186/"

# ── Relancer uniquement l'extraction des variables (ex: après édition config) ─
python ~/digitize_medical_records/SystemB/VariableExtraction/extract_variables.py \
  --all \
  --config ~/digitize_medical_records/SystemB/VariableExtraction/variable_extraction_config.yaml \
  --output-dir ~/digitize_medical_records/SystemB/VariableExtraction/outputs

# ── Lancer le pipeline sans générer Excel/SQLite ─────────────────────────────
bash ~/digitize_medical_records/SystemB/run_pipeline.sh \
  "/chemin/vers/patients/" --skip-variables
```

---

## Préparer vos Documents Patients

Organisez vos fichiers PDF avec **un dossier par patient** :

```
mes_donnees_patients/
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

Les noms de dossiers peuvent être quelconques — le système les utilise comme identifiants patients.
Les données peuvent se trouver **n'importe où sur votre serveur** — elles n'ont pas besoin d'être dans le dossier `digitize_medical_records`.

---

## Ce Qui Se Passe Étape par Étape

```
ENTRÉE : Dossier patient (ex : "2006 RDB 0186/")
  └── Contient : DOC_00118.pdf, DOC_00119.pdf, DOC_00122.pdf ...

ÉTAPE 1 : GLM-OCR
  Chaque PDF est lu page par page par GLM-OCR.
  Produit : texte brut + images de pages.
  Les documents déjà traités sont ignorés automatiquement.

ÉTAPE 2 : Classification des pages (Qwen 7B)
  Chaque image de page est montrée à Qwen 7B qui décide :
  "Est-ce un formulaire CNR Paludisme ou un document clinique ?"

ÉTAPE 3 : Extraction
  ├── Pages CNR    → Qwen 72B extrait tous les champs du formulaire
  │                  (espèce, dates, traitement, valeurs biologiques, etc.)
  │                  Puis postprocess_checker.py valide la sortie.
  └── Pages non-CNR → texte GLM-OCR uniquement
                      (résultats de laboratoire, comptes rendus cliniques, urgences)

ÉTAPE 4 : Fusion
  Toutes les sorties de documents fusionnées en un seul fichier par patient.
  Sortie : outputs/patients/RDB_0186/RDB_0186_patient_raw.txt

ÉTAPE 5 : Anonymisation
  Tous les identifiants personnels sont supprimés :
    - Nom du patient  → [PATIENT_001]
    - Noms des médecins → [STAFF_001], [STAFF_002] ...
    - Téléphone, email, adresse, NPI, FINESS → [ANONYMIZED]
  Utilise Qwen 7B NER (local, sans internet) pour détecter les noms.
  Chaque dossier patient reçoit un ID séquentiel permanent stocké dans
  outputs/patient_registry.json — retraiter le même dossier réutilise
  toujours le même ID, sans jamais créer de doublons.
  Sorties :
    outputs/patients/RDB_0186/RDB_0186_patient_anonymized.txt
    outputs/patients/RDB_0186/RDB_0186_replacements.csv
    outputs/final_anonymized/patient_001_anonymized.txt

ÉTAPE 6 : Extraction des Variables
  S'exécute UNE FOIS à la fin pour TOUS les patients ensemble.
  Assigne chaque document/page à J0, J3 ou J30 :
    - J0  = première date de consultation (admission)
    - J3  = première date de suivi au-delà de 3 jours depuis J0
    - J30 = dernière date de consultation
    - Les dates dans les 3 jours suivant J0 sont fusionnées dans J0
  Extrait 130 variables cliniques par patient.
  Écrit une ligne par patient dans :
    SystemB/VariableExtraction/outputs/research_table.xlsx
    SystemB/VariableExtraction/outputs/research_database.db
```

---

## Fichiers de Sortie

```
digitize_medical_records/
├── outputs/
│   ├── patient_registry.json             ← mapping permanent dossier → ID
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
            ├── research_table.xlsx    ← UN seul fichier, tous les patients
            └── research_database.db  ← UNE seule base de données
```

> **Important :** Chaque nouveau patient est **ajouté** au même fichier Excel et
> à la même base SQLite. Traiter 15 patients donne 15 lignes dans une seule feuille.
> Retraiter le même patient **met à jour** sa ligne — aucun doublon.

---

## Consulter les Résultats

### Excel
Télécharger sur votre machine locale :
```bash
# Exécuter sur votre machine LOCALE :
scp utilisateur@votre-serveur:~/digitize_medical_records/SystemB/VariableExtraction/outputs/research_table.xlsx ~/Desktop/
```
Puis ouvrir dans Excel ou LibreOffice.

Ou visualiser directement dans VSCode avec l'extension **Excel Viewer** (MESCIUS/GrapeCity).

### SQLite
Visualiser dans VSCode avec l'extension **SQLite Viewer** (Florian Klampfer) — cliquez simplement sur le fichier `.db`.

Ou interroger depuis le terminal :
```bash
sqlite3 ~/digitize_medical_records/SystemB/VariableExtraction/outputs/research_database.db \
  "SELECT ID_Patient, Sexe, Age, gravite_palu, hemoglobine_J0 FROM patients;"

sqlite3 ~/digitize_medical_records/SystemB/VariableExtraction/outputs/research_database.db \
  "SELECT COUNT(*) FROM patients;"
```

---

## Système d'Identification des Patients

Chaque dossier patient reçoit un **ID séquentiel permanent** lors de son premier traitement.
Ce mapping est sauvegardé dans `outputs/patient_registry.json` :

```json
{
  "RDB_0186": "001",
  "RDB_0201": "002",
  "RDB_0189": "003"
}
```

- Retraiter le même dossier réutilise toujours le même ID
- Les nouveaux dossiers reçoivent automatiquement le numéro suivant
- Aucun doublon, aucune gestion manuelle nécessaire

---

## Confidentialité et Sécurité des Données

| Composant | Accès réseau | Données envoyées à l'extérieur |
|---|---|---|
| GLM-OCR | Ollama local uniquement | Jamais |
| Qwen 7B (classification + anonymisation) | Ollama local uniquement | Jamais |
| Qwen 72B (extraction formulaires CNR) | Ollama local uniquement | Jamais |
| Fichier Excel | Fichier local | Jamais |
| Base de données SQLite | Fichier local | Jamais |

**Aucune donnée patient ne quitte votre serveur à aucun moment.**

---

## Adapter le Système à une Nouvelle Institution

Modifier uniquement **un fichier** : `SystemB/VariableExtraction/variable_extraction_config.yaml`

Changer uniquement ces 4 sections en haut du fichier :

```yaml
institution:
  services:
    - "NOM DE VOTRE SERVICE"    # ex : "Pédiatrie", "Urgences"

lieu_sejour_keywords:            # mots-clés destination de voyage → noms standardisés
  "mot-clé pays": "Nom Standardisé du Pays"

chimio_keywords:                 # noms des médicaments prophylactiques
  "mot-clé médicament": "Nom Standardisé"

treatment_keywords:              # noms des médicaments de traitement
  "mot-clé médicament": "Nom Standardisé"
```

Tout le reste — OCR, classification, anonymisation, logique J0/J3/J30,
formatage Excel, schéma SQLite — fonctionne sans aucune modification.

> **Note sur les accents français :** L'OCR supprime parfois les accents
> (ex : `Symptômes` → `Symptomes`, `Consuliation` au lieu de `Consultation`).
> Le pipeline gère cela automatiquement grâce à la normalisation des accents.
> Aucune correction manuelle n'est nécessaire.

---

## Limitations Connues

- `fièvre_J0` indique si la fièvre a été documentée dans les notes cliniques, pas nécessairement le motif principal de consultation
- Date J3 = première date de suivi au-delà de 3 jours depuis l'admission (peut varier de ±1 jour selon les dates des documents)
- Les variables restent vides si elles ne sont pas documentées dans les dossiers scannés
- Les champs du formulaire CNR ont une priorité inférieure aux valeurs des documents cliniques (configurable par champ)

---

## Résolution de Problèmes

**Un patient est absent de la base de données :**
```bash
python ~/digitize_medical_records/SystemB/VariableExtraction/extract_variables.py \
  --all \
  --config ~/digitize_medical_records/SystemB/VariableExtraction/variable_extraction_config.yaml \
  --output-dir ~/digitize_medical_records/SystemB/VariableExtraction/outputs
```

**Vérifier quel dossier correspond à quel ID patient :**
```bash
cat ~/digitize_medical_records/outputs/patient_registry.json
```

**Le pipeline échoue sur un patient — vérifier ce qui a été extrait :**
```bash
cat ~/digitize_medical_records/outputs/patients/RDB_XXXX/RDB_XXXX_patient_raw.txt | head -50
```

**L'extraction de variables manque une valeur :**
Modifier `SystemB/VariableExtraction/variable_extraction_config.yaml` dans la section `fields:`.
Aucune modification de code Python nécessaire. Puis relancer l'extraction avec `--all`.

**Ollama ne fonctionne pas :**
```bash
ollama serve &
sleep 3
ollama list
```

---

## Structure du Dépôt

```
paludisme/
├── SystemA/                          ← Système A (pipeline précédent)
├── SystemB/                          ← Système B (ce pipeline)
│   ├── run_pipeline.sh               ← POINT D'ENTRÉE PRINCIPAL
│   ├── Anonymization_systemB/        ← module d'anonymisation
│   ├── CNR_forms/                    ← scripts OCR + extraction CNR
│   ├── SystemB_page_classification/  ← classificateur de pages + orchestrateur
│   └── VariableExtraction/           ← extracteur de 130 variables + config
├── .gitignore
└── requirements.txt
```

---

## Contact et Support

Développé par **Labiba FAROOQ** à **ObTIC, Sorbonne Université**.

GitHub : https://github.com/obtic-sorbonne/paludisme
