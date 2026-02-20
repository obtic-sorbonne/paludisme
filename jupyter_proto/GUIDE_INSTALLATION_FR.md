# GUIDE D'INSTALLATION ET D'UTILISATION

---

## ÉTAPE 1 : INSTALLER PYTHON

## ÉTAPE 2 : INSTALLER TESSERACT OCR

### 2.1 Télécharger Tesseract

1. Aller sur : **https://github.com/UB-Mannheim/tesseract/wiki**
2. Cliquer sur **"tesseract-ocr-w64-setup-5.x.x.xxxxxxxx.exe"** (version la plus récente)
3. Le fichier se télécharge

### 2.2 Installer Tesseract

1. **Double-cliquer** sur le fichier téléchargé
2. Cliquer sur **"Next"** plusieurs fois
3. ⚠️ **IMPORTANT** : Quand vous voyez "Choose Components" :
   - Cocher **"Additional language data"**
   - Dans la liste qui apparaît, cocher **"French"**
4. Continuer avec **"Next"** puis **"Install"**
5. Noter le chemin d'installation (par défaut : `C:\Program Files\Tesseract-OCR`)
6. Cliquer sur **"Finish"**

### 2.3 Vérifier l'installation

1. Ouvrir l'**Explorateur de fichiers**
2. Aller dans : `C:\Program Files\Tesseract-OCR`
3. Vous devez voir un fichier `tesseract.exe`
4. Aller dans le sous-dossier `tessdata`
5. Vous devez voir un fichier `fra.traineddata`

✅ Si vous voyez ces fichiers, c'est bon !

## ÉTAPE 3 : INSTALLER JUPYTER NOTEBOOK

## ÉTAPE 4 : CLONER LE REPO GIT ET LANCER LE NOTEBOOK

### 4.1 Cloner le répo

### 4.2 Mettre le dossier avec les données dans le même répo


### 4.3 Ouvrir le notebook

## ÉTAPE 5 : CONFIGURER LE NOTEBOOK 

### 5.1 Modifier Cell 1

Dans la **Cell 1**, ligne 3, modifier le chemin vers Tesseract :

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\TON CHEMIN VERS LES DOSSIERS SUIVANTS\Tesseract-OCR\tesseract.exe'
```

### 5.2 Modifier Cell 3

Dans la **Cell 3**, ligne 10, modifier le chemin vers tes PDFs :

```python
root_directory = r"C:\Users\VotreNom\Documents\mes_pdfs_medicaux"
```

Remplacer par le vrai chemin où se trouvent vos dossiers de patients.

---

## ÉTAPE 6 : EXÉCUTER LE NOTEBOOK


### 6.2 Pendant le traitement


**Si le système demande un nom manuellement :**

Cela signifie que le système n'a pas reconnu le nom et le prénom du patient et utilise un fallback. 


```
⚠ MANUAL INPUT NEEDED
====================================================================
Folder: 2007 RDB 0071
PDF files in folder:
  - 2007 RDB 0071.pdf
====================================================================
  LASTNAME (multi-word OK): 
```

Tapez le nom de famille (en MAJUSCULES), puis Entrée
Tapez le prénom (en MAJUSCULES), puis Entrée


## ÉTAPE 7 : RÉCUPÉRER LES RÉSULTATS

### 7.1 Localiser les fichiers


Structure :
```
pipeline_output/
├── anonymized_texts/          ← Vos fichiers anonymisés
│   ├── PATIENT_001_foldername.txt
│   ├── PATIENT_002_foldername.txt
│   └── ...
├── patient_mapping.csv        ← ⚠️ CONFIDENTIEL - Correspondance codes/noms
└── patient_mapping.json
```

### 7.2 Vérifier la qualité

Ouvrir les fichiers .txt et vérifier :

✅ Les noms de patients sont remplacés par `PATIENT_XXX`
✅ Les données personnelles sont complètement éliminées
