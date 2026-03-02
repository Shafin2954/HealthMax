# HealthMax — Dataset Download Instructions

Download all datasets **before** writing any pipeline code.
Place all raw files in `data/raw/` (this folder is `.gitignore`d — do NOT commit raw data).

---

## Priority 1 — Core Engine Datasets (Download First)

| # | Dataset | Source | Target filename in `data/raw/` |
|---|---------|--------|-------------------------------|
| 1 | Bangla Symptoms-Disease Dataset | [arXiv:2601.12068](https://arxiv.org/abs/2601.12068) + Mendeley | `bangla_symptoms_disease.csv` |
| 2 | Bengali Medical NER Dataset | [Kaggle / Mendeley: 4tt953xwk2.1](https://data.mendeley.com/datasets/4tt953xwk2/1) | `bangla_health_ner.csv` |
| 3 | Bangla MedER Dataset | [Mendeley: BanglaHealthNER](https://data.mendeley.com/) | `bangla_meder.csv` |
| 4 | Bangla HealthNER Corpus | [GitHub search: BanglaHealthNER](https://github.com/) | `bangla_health_ner_corpus/` |

## Priority 2 — Drug Data

| # | Dataset | Source | Target filename |
|---|---------|--------|-----------------|
| 5 | Bangladesh DGDA Medicine Registry | [Mendeley: 3x5gsr2jm3.1](https://data.mendeley.com/datasets/3x5gsr2jm3/1) | `dgda_medicine_registry.csv` |

## Priority 3 — ASR & Speech

| # | Dataset | Source | Target |
|---|---------|--------|--------|
| 6 | OpenSLR Bangla Speech | [openslr.org/53](https://openslr.org/53/) | `openslr_bangla/` |
| 7 | Mozilla Common Voice 17 (Bangla) | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/en/datasets) | `cv17_bn/` |

## Priority 4 — Dialect

| # | Dataset | Source | Target |
|---|---------|--------|--------|
| 8 | BanglaDialecto Corpus | GitHub | `bangla_dialecto/` |
| 9 | Bhasha Bichitra | Kaggle | `bhasha_bichitra/` |

---

## After Downloading

Run the data processor to build all model artifacts:

```bash
cd data/
python process_datasets.py --step all
```

This will produce:
- `models/disease_rag.index`
- `models/disease_records.json`
- `models/symptom_binarizer.json`
- `models/disease_label_encoder.json`
- `models/disease_classifier.json`
- `data/dgda_medicines_clean.csv`

---

## Notes

- `data/raw/` is in `.gitignore` — never commit raw datasets to the repo.
- If a dataset URL is dead, check the Mendeley DOI directly or contact the dataset owner.
- For NER fine-tuning, use Google Colab — see `notebooks/banglabert_finetune.ipynb`.
