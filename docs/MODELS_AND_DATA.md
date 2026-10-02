# Models and data

Every number on this page comes from a file in the repo, named next to it.

## Datasets

| Dataset | File(s) | Source | Used for |
|---|---|---|---|
| Bangla symptoms–disease dataset: 757 rows, 85 diseases | `data/raw/Symptoms.csv` (copy in `assets/`) | Zannat, Al Shafi & Muntakim, *Bridging the Gap in Bangla Healthcare*, ECCE 2025 / arXiv:2601.12068 ([Mendeley](https://data.mendeley.com/datasets/rjgjh8hgrt/6)). The paper is in `assets/` | Classifier, retrieval records, symptom vocabulary |
| DGDA medicine list: ~21.7k brands with generic name, strength and price | `assets/medicine.csv` | Bangladesh DGDA registry (see `data/README.md`) | Medicine lookup |
| Medicine description texts (Bangla) | `healthmax-ai-assistant/src/data/medicine_ner.csv`, `medicine_ner_v2.csv` | Collected for the project | Silver NER data |
| Specialist classification (problem → specialist) | `healthmax-ai-assistant/src/data/specialist_classification.csv` | Collected for the project | Silver NER data, specialist mapping |
| Candidate sources that were listed but not used | `assets/Sources.md` | Mendeley (BNDSNER, MedBanglaTrust3, MedER), OpenSLR 37 | Not used |

Caveats:
- `healthmax-ai-assistant/src/data/Symptoms.csv` and `Symptom.csv` are **Excel workbooks saved with a `.csv` name**, so the pipeline ignores them.
- The symptoms–disease dataset records only **which symptoms are present**. It has no age, sex, duration, severity or prevalence. That is the main limit on what any model trained on it can learn.

## Models

### Bangla medical NER (BanglaBERT)

- Base model: `sagorsarker/bangla-bert-base`. Fine-tuned on an RTX 3070 Ti for 4 epochs in about 2 minutes (`models/ner_training_summary.json`).
- Tags: SYMPTOM, DISEASE and MEDICINE, in BIO format.
- Data: **3,167 silver-labelled sentences** (2,850 train / 317 validation), built automatically by `data/build_ner_dataset.py` from the sources above:

  | Source | Sentences |
  |---|---|
  | Medicine texts | 2,385 |
  | Synthetic symptom sentences | 641 |
  | Specialist problems | 141 |

- Validation scores:

  | Metric | Score |
  |---|---|
  | F1 | **0.790** |
  | Precision | 0.763 |
  | Recall | 0.818 |
  | Token accuracy | 0.954 |

- **Caveat:** the labels are dictionary matches, not human annotations. For example, চুলকানি (itching) is tagged DISEASE in some sentences. So the score measures agreement with the labelling rules, not with medical reality.

### Disease classifier (XGBoost)

- Input: 166 binary symptom features. Output: 85 diseases.
- Split: 605 train / 152 test, stratified (`models/training_summary.json`).
- **Macro F1: 0.726** on the held-out rows.
- With about 9 rows per disease, the test set has only one or two rows per class, so this number is noisy.
- The dataset paper reports 98% accuracy with a voting ensemble on the same data, which mostly shows how easy the dataset's own rows are, not how the model handles real patients.

### Retrieval

- The 85 disease records (`models/disease_records.json`) each hold symptoms, urgency, specialist and care level.
- Shipped backend: **TF-IDF** (`rag_backend: "tfidf"` in `models/training_summary.json`). The sentence-transformer + FAISS path exists in `backend/rag.py` but wasn't used for the shipped artifacts.
- Urgency labels in the records: **82 URGENT, 3 EMERGENCY, 0 SELF-CARE**. The [evaluation](EVALUATION.md) shows how that skews the output.

### Speech

- Speech to text: `asif00/whisper-bangla` on the server; the browser's Web Speech API (`bn-BD`) in the web app.
- Text to speech: Google Cloud TTS (`backend/tts.py`). Optional, and needs credentials.

## Reproducing

```powershell
conda env create -f environment.yml      # or: py -3.12 -m venv .venv; pip install -r requirements.txt
python data/process_datasets.py          # classifier + retrieval artifacts -> models/
python data/build_ner_dataset.py         # silver NER dataset -> data/processed/
python training/train_ner.py             # BanglaBERT fine-tune -> models/ner-banglabert-medical/
python scripts/export_browser_artifacts.py   # JSON artifacts for the web app -> healthmax-ai-assistant/public/model/
```

Large binaries (`*.safetensors`, `disease_classifier.json`, FAISS index) are gitignored. Run the steps above to regenerate them.
