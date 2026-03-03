# HealthMax — Dataset Download Guide

Download ALL datasets before running `process_datasets.py`.
Save them to `data/raw/` with the filename shown.

## Priority 1 — Core Engine (Download First)

| # | Dataset | Source | Save As |
|---|---------|--------|---------|
| 1 | Bangla Symptoms-Disease | [Mendeley](https://data.mendeley.com/datasets/) arXiv:2601.12068 | `symptoms_disease_bangla.csv` |
| 2 | Bengali Medical NER | [Kaggle](https://www.kaggle.com/) Mendeley: 4tt953xwk2.1 | `bengali_medical_ner.json` |
| 3 | Bangla MedER Dataset | Mendeley: BanglaHealthNER | `bangla_meder.json` |
| 4 | Bangla HealthNER Corpus | GitHub: BanglaHealthNER | `bangla_healthner.json` |

## Priority 2 — Drug Data

| # | Dataset | Source | Save As |
|---|---------|--------|---------|
| 5 | DGDA Medicine Registry | Mendeley: 3x5gsr2jm3.1 | `dgda_medicines.csv` |

## Priority 3 — Speech (for ASR fine-tuning)

| # | Dataset | Source |
|---|---------|--------|
| 6 | OpenSLR Bangla Speech | openslr.org/37 |
| 7 | Mozilla Common Voice Bangla | commonvoice.mozilla.org |

## Priority 4 — Dialects

| # | Dataset | Source |
|---|---------|--------|
| 8 | BanglaDialecto Corpus | GitHub search: BanglaDialecto |

> **Note:** The `data/raw/` directory is in `.gitignore`. Never commit raw datasets.
