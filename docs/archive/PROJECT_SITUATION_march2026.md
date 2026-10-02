# HealthMax Project Situation

Date: 2026-03-25

## Current reality

HealthMax is no longer just a mockup plus plan. The repo now has a working **local end-to-end path**:

1. user enters Bangla symptoms in the Lovable app
2. the Lovable `/triage` page sends a request to the local FastAPI backend
3. the backend runs:
   - Bangla medical NER
   - disease retrieval / RAG
   - XGBoost disease classification
   - clinical safety rules
   - DGDA medicine lookup
4. the app renders the returned disease list, urgency, facility, and medicines

This path was browser-verified locally on 2026-03-25.

The current priority is **quality improvement and hosted integration**, not basic wiring.

AWS is still deferred.

## Verified architecture

### Local path now working

`healthmax-ai-assistant/src/pages/Triage.tsx`
-> `VITE_HEALTHMAX_API_URL`
-> local FastAPI `/api/triage`
-> `backend/ner.py`
-> `backend/rag.py`
-> `backend/classifier.py`
-> `backend/rules.py`
-> `backend/dgda_lookup.py`
-> structured JSON response
-> Lovable UI

### Hosted path not finished yet

`healthmax-ai-assistant`
-> Supabase Edge Function `healthmax-triage`
-> configurable backend URL
-> hosted Lovable app / WhatsApp / voice flows

This hosted path is partly prepared, but the local path is the only path fully verified end to end right now.

## Canonical data files in use

### 1. Disease classifier and disease retrieval

Primary dataset:

- `data/raw/Symptoms.csv`

Current use:

- trains the XGBoost disease classifier
- builds disease retrieval records
- builds the FAISS index
- supplies canonical symptom and disease surfaces

Important facts:

- 757 rows
- 85 diseases
- 166 symptom features in the current processed artifact set

### 2. Medicine lookup

Primary dataset:

- `assets/medicine.csv`

Current use:

- DGDA medicine lookup
- generic / brand / price suggestions

This is a lookup dataset, not a training dataset.

### 3. Specialist routing

Current dataset:

- `healthmax-ai-assistant/src/data/specialist_classification.csv`

Current use:

- usable as a future specialist classifier dataset
- not yet trained into a dedicated local model

### 4. NER silver-label source data

Current source files used to build the local NER dataset:

- `data/raw/Symptoms.csv`
- `healthmax-ai-assistant/src/data/medicine_ner_v2.csv`
- `healthmax-ai-assistant/src/data/medicine_ner.csv`
- `healthmax-ai-assistant/src/data/specialist_classification.csv`

These are not a gold BIO corpus. They are used to build a **silver-labeled** NER training dataset.

## Models currently trained

### Disease classifier

Training source:

- `data/raw/Symptoms.csv`

Artifacts:

- `models/disease_classifier.json`
- `models/label_encoder.json`
- `models/symptom_list.json`

Current metrics from `models/training_summary.json`:

- macro F1: `0.7255`
- rows: `757`
- diseases: `85`
- symptom features: `166`
- split: `605 train / 152 test`

### Disease retrieval / RAG

Training source:

- `data/raw/Symptoms.csv`

Artifacts:

- `models/disease_rag.index`
- `models/disease_records.json`
- `models/rag_config.json`

Current runtime:

- sentence-transformer embeddings
- embedding model: `paraphrase-multilingual-MiniLM-L12-v2`
- embeddings on CUDA
- FAISS on CPU

Notes:

- FAISS GPU is still not available in this Windows + Python 3.12 setup
- XGBoost and embedding generation do use the RTX 3070 Ti

### Bangla medical NER

Base model:

- `sagorsarker/bangla-bert-base`

Fine-tuned local output:

- `models/ner-banglabert-medical/`
- `models/ner_training_summary.json`

Silver dataset output:

- `data/processed/ner_silver_train.jsonl`
- `data/processed/ner_silver_validation.jsonl`
- `data/processed/ner_silver_summary.json`

Current NER metrics:

- validation F1: `0.7896`
- precision: `0.7635`
- recall: `0.8175`
- accuracy: `0.9542`
- examples: `3167 total`, `2850 train`, `317 validation`

Important limitation:

- this is trained on silver labels, not medically reviewed gold BIO annotations

## What is working now

### Backend

Working:

- FastAPI `/health`
- FastAPI `/api/triage`
- Bangla symptom extraction
- disease-name mention detection
- medicine mention detection
- classifier predictions
- RAG retrieval
- emergency override rules
- DGDA medicine lookup

### Lovable app

Working locally:

- `healthmax-ai-assistant/.env` points to `http://127.0.0.1:8000`
- the Lovable app posts directly to the local backend
- browser verification confirmed `POST http://127.0.0.1:8000/api/triage`
- model-backed response is rendered in the app

### Quality improvements already made

- symptom alias normalization improved
- hybrid ranking added:
  - classifier probability
  - RAG similarity
  - symptom overlap
  - exact disease-name boosting
- medicine-only input now avoids inventing a fake disease list

## What is still weak

### Main quality gap

Ambiguous fever-like symptom sets are still not strong enough.

Example:

- dengue-like prompts now surface `ডেঙ্গু`
- but `ম্যালেরিয়া` can still rank above it in some cases

So the core remaining problem is **ranking quality**, not app wiring.

### Not finished yet

- gold medical NER corpus
- specialist classifier
- Supabase imports and hosted validation
- WhatsApp / voice end-to-end hosted tests
- ASR fine-tuning
- future public hosting choice, if needed

### External-service limitation

If OpenAI / Bedrock credentials are not set, response generation falls back to the template path.

## Decisions already made

- `data/raw/Symptoms.csv` is the canonical local disease dataset
- `assets/medicine.csv` is the canonical local medicine lookup dataset
- local testing comes before hosted integration
- any public hosting choice comes after local + hosted stability
- NER is allowed to use a silver-label interim dataset for now

## Best next direction

### Immediate

1. improve disease ranking quality
2. create a repeatable evaluation set for common Bangla cases
3. import datasets into Supabase
4. test the Lovable hosted app against a reachable backend URL

### After that

1. replace silver NER data with reviewed gold labels
2. train specialist routing model
3. stabilize voice / WhatsApp path
4. then decide whether a public hosting layer is even needed

## Bottom line

HealthMax is now in this state:

- local model training works
- local backend inference works
- local Lovable app integration works
- the backend is using real trained artifacts
- the biggest remaining risk is prediction quality, not missing infrastructure
