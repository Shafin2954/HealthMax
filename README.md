# HealthMax

**Bangla AI Health Triage System** for symptom-based triage, medicine lookup, and facility guidance.

## Current state

HealthMax has two parallel triage paths:

- the FastAPI backend (`backend/`), which runs NER, disease retrieval,
  classification, rules, and medicine lookup, and returns structured
  Bangla triage results
- the Lovable app's own **client-side** triage engine
  (`healthmax-ai-assistant/src/lib/browserTriage.ts`), a TypeScript
  reimplementation of the same NER/RAG/classifier/rules pipeline that
  runs entirely in the browser

As of the current `Triage.tsx`, the `/triage` page calls
`runBrowserTriage()` directly — it does **not** call the FastAPI
backend. The two pipelines are logically equivalent but are
maintained as separate codebases, so a fix made in one (e.g. a rules
bug) must be mirrored in the other.

What is **not** finished yet:

- hosted Supabase validation
- WhatsApp / voice end-to-end validation
- future public hosting choice, if needed
- gold-label NER data
- higher-quality disease ranking on ambiguous fever cases
- reconciling the backend and browser triage pipelines into one path

## What works now

### Local backend

- `/health`
- `/api/triage`
- Bangla symptom extraction
- disease mention detection
- medicine mention detection
- emergency rule override
- DGDA medicine lookup

### Local Lovable app

- the Lovable `/triage` page runs `runBrowserTriage()` client-side
- model-backed results render in the browser
- the app can show:
  - top diseases
  - urgency
  - facility recommendation
  - medicine suggestions

## Canonical datasets

### Disease classifier and disease retrieval

- `data/raw/Symptoms.csv`

Used for:

- XGBoost disease classifier
- disease retrieval records
- FAISS index

### Medicine lookup

- `assets/medicine.csv`

Used for:

- DGDA brand / generic / price lookup

### Specialist routing

- `healthmax-ai-assistant/src/data/specialist_classification.csv`

Used for:

- future specialist-routing model work

### NER silver-label sources

- `data/raw/Symptoms.csv`
- `healthmax-ai-assistant/src/data/medicine_ner_v2.csv`
- `healthmax-ai-assistant/src/data/medicine_ner.csv`
- `healthmax-ai-assistant/src/data/specialist_classification.csv`

Used for:

- local silver-labeled Bangla medical NER training

## Current trained artifacts and scores

### Disease classifier

Artifacts:

- `models/disease_classifier.json`
- `models/label_encoder.json`
- `models/symptom_list.json`

Score:

- macro F1: `0.7255`

### Disease retrieval / RAG

Artifacts:

- `models/disease_rag.index`
- `models/disease_records.json`
- `models/rag_config.json`

Runtime:

- sentence-transformer embeddings on CUDA
- FAISS on CPU

### Bangla medical NER

Artifacts:

- `models/ner-banglabert-medical/`
- `models/ner_training_summary.json`

Score:

- validation F1: `0.7896`

## Repository structure

```text
HealthMax/
├── backend/                     # FastAPI inference pipeline
├── data/                        # training pipeline and raw/processed datasets
├── models/                      # trained local artifacts
├── training/                    # standalone model training scripts
├── notebooks/                   # notebook experiments
├── frontend/                    # legacy static demo
├── healthmax-ai-assistant/      # Lovable app + Supabase functions
├── PROJECT_SITUATION.md         # current repo/project reality
├── tasks.md                     # current worklist
└── README.md
```

## Local run

### Python version first

Use `Python 3.12` for this pinned backend stack.

If you try `Python 3.13`, install can fail on `torch==2.3.0` and possibly other pinned ML wheels.

Recommended Windows setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Start the backend (optional — the Lovable app no longer calls it)

From the repo root:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 2. Start the Lovable app

From `healthmax-ai-assistant`:

```powershell
npm run dev
```

The dev server runs on port `8080` (set in `vite.config.ts`), not Vite's
default `5173`.

### 3. Open in browser

- backend health (if running): `http://127.0.0.1:8000/health`
- Lovable triage page: `http://127.0.0.1:8080/triage`

If you run preview instead of dev, Vite still serves it on `8080`
unless overridden.

### 4. Verify the triage pipeline

The `/triage` page runs `runBrowserTriage()` from
`src/lib/browserTriage.ts` entirely client-side — no network call to
the FastAPI backend is involved. Submit a triage prompt and check the
rendered result contains:

- matched symptoms/diseases
- `urgency_level`
- a facility recommendation
- medicine suggestions

To exercise the FastAPI backend's equivalent pipeline instead (e.g.
for the WhatsApp/voice paths, which do use it), call
`POST http://127.0.0.1:8000/api/triage` directly and check for
`ner_entities`, `top_diseases`, `urgency_level`, `medicines`.

## Example manual test prompts

- `বুকে ব্যথা হচ্ছে এবং শ্বাস নিতে কষ্ট হচ্ছে।`
  Expected: emergency guidance

- `আমার ডায়রিয়া, পেট ব্যথা আর পানিশূন্যতা হচ্ছে।`
  Expected: urgent gastro/cholera-like ranking

- `আমার বাচ্চার হাম হয়েছে মনে হচ্ছে।`
  Expected: `হাম` near the top

- `আমার কয়েকদিন ধরে জ্বর, চোখের পেছনে ব্যথা, গায়ে ব্যথা আর বমি বমি লাগছে।`
  Expected: `ডেঙ্গু` and/or `ম্যালেরিয়া` near the top

## Collaborator guide

The project is no longer at the “implement the skeleton” stage. The main work now is **quality, evaluation, and hosted integration**.

### Workstream A: Disease ranking quality

Needed:

- improve dengue / malaria / flu-like separation
- tune hybrid ranking weights
- improve Bangla symptom normalization
- reduce weak tied predictions in low-signal cases

Best files:

- `backend/fusion.py`
- `backend/classifier.py`
- `backend/rag.py`
- `backend/main.py`

### Workstream B: Evaluation and benchmarks

Needed:

- build a saved benchmark prompt set
- define expected top-3 results for common cases
- add regression checks so ranking quality does not drift

Best files:

- `tasks.md`
- `models/training_summary.json`
- new files under `tests/`

### Workstream C: NER quality

Needed:

- replace silver labels with a reviewed gold BIO dataset
- improve per-entity quality for symptom / disease / medicine tags
- add stronger NER evaluation by entity type

Best files:

- `data/build_ner_dataset.py`
- `training/train_ner.py`
- `backend/ner.py`
- `notebooks/banglabert_finetune.ipynb`

### Workstream D: Supabase and hosted integration

Needed:

- import datasets into Supabase tables
- point hosted Edge Functions to a reachable backend URL
- validate the hosted Lovable app path

Best files:

- `healthmax-ai-assistant/src/pages/Triage.tsx`
- `healthmax-ai-assistant/supabase/functions/healthmax-triage/index.ts`
- `healthmax-ai-assistant/src/pages/AdminImport.tsx`

### Workstream E: Voice and messaging

Needed:

- stabilize Twilio WhatsApp flow
- validate voice path against the current backend
- decide when ASR becomes a true training priority

Best files:

- `backend/asr.py`
- `backend/main.py`
- `healthmax-ai-assistant/supabase/functions/twilio-whatsapp/index.ts`
- `healthmax-ai-assistant/supabase/functions/twilio-voice/index.ts`

## Collaboration rules

- Do not weaken emergency rules without explicit review
- Keep the local model-to-app path working
- Prefer canonical datasets already in use
- Record metric changes when retraining models
- Do not assume AWS is the next step; local quality and hosted validation come first, and any public hosting choice stays future-only for now

## Important files to read first

- `PROJECT_SITUATION.md`
- `tasks.md`
- `backend/main.py`
- `backend/fusion.py`
- `backend/ner.py`
- `data/process_datasets.py`

## Bottom line

HealthMax is now a working local prototype with real trained artifacts and real browser integration.

The next milestone is:

- better prediction quality
- stronger evaluation
- hosted Supabase/Lovable validation
