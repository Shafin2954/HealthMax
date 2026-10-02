# HealthMax Tasks

## Goal

Keep the local model-to-app path working, improve prediction quality, then move the same backend path into Supabase-hosted usage.

## Current status

- [x] Local training pipeline works
- [x] Disease classifier is trained from `data/raw/Symptoms.csv`
- [x] Disease RAG index is built from the same dataset
- [x] BanglaBERT NER is fine-tuned locally on a silver medical dataset
- [x] FastAPI serves real trained artifacts
- [x] Lovable `/triage` fetches model-backed data from `http://127.0.0.1:8000/api/triage`
- [x] Browser-level local verification is complete
- [ ] Prediction quality is strong enough for common ambiguous fever cases
- [ ] Hosted Supabase / Lovable integration is fully validated
- [ ] Voice / WhatsApp flow is fully validated

## Current scores

### Disease classifier

- macro F1: `0.7255`
- rows: `757`
- diseases: `85`
- features: `166`

### NER

- validation F1: `0.7896`
- precision: `0.7635`
- recall: `0.8175`
- accuracy: `0.9542`

## Phase 0: Environment and runtime

- [x] Repair local Python / torch runtime
- [x] Confirm CUDA on local RTX 3070 Ti
- [x] Install backend requirements
- [x] Configure Lovable local backend URL
- [ ] Create a clean reproducible venv setup note for collaborators
- [ ] Create a backend `.env` with all optional keys documented

## Phase 1: Canonical data

- [x] Use `data/raw/Symptoms.csv` as canonical disease dataset
- [x] Use `assets/medicine.csv` as canonical local medicine lookup dataset
- [x] Use `healthmax-ai-assistant/src/data/specialist_classification.csv` as future specialist-routing dataset
- [x] Use `medicine_ner_v2.csv` / `medicine_ner.csv` as interim silver-label NER source data
- [x] Ignore broken Excel-as-CSV legacy files in the main local pipeline

## Phase 2: Local trained artifacts

- [x] Train classifier artifacts into `models/`
- [x] Build disease retrieval artifacts into `models/`
- [x] Save training summaries
- [x] Fine-tune and save local NER model
- [ ] Add one command or script that reruns all local training steps in order

## Phase 3: Backend quality

- [x] Clean `backend/classifier.py`
- [x] Clean `backend/rag.py`
- [x] Clean `backend/rules.py`
- [x] Clean `backend/main.py`
- [x] Clean `backend/dgda_lookup.py`
- [x] Add hybrid ranking:
  - classifier score
  - RAG score
  - symptom overlap
  - disease mention boost
- [x] Improve symptom alias handling
- [ ] Tune hybrid ranking weights for dengue / malaria / flu-like overlaps
- [ ] Expand symptom normalization coverage for more Bangla phrasing
- [ ] Reduce low-value tied disease lists in weak-signal cases

## Phase 4: Local validation

- [x] Test `/health`
- [x] Test `/api/triage` with Bangla symptom inputs
- [x] Test emergency rule override
- [x] Test medicine-name-only input
- [x] Test disease-name-only input
- [x] Test Lovable app locally in browser
- [ ] Create a saved local benchmark set with expected top-3 disease behavior
- [ ] Add repeatable regression checks for the benchmark prompts

## Phase 5: NER follow-up

- [x] Build local silver-labeled NER dataset
- [x] Fine-tune BanglaBERT locally
- [x] Wire backend to prefer local fine-tuned NER checkpoint
- [ ] Replace silver labels with reviewed gold BIO annotations
- [ ] Add per-entity evaluation by label:
  - symptom
  - disease
  - medicine
- [ ] Improve short-phrase disease extraction beyond lexicon-only coverage

## Phase 6: Specialist and recommendation layer

- [ ] Train a specialist-routing model from `specialist_classification.csv`
- [ ] Compare trained specialist routing vs current lookup-based routing
- [ ] Feed specialist output cleanly into the triage response

## Phase 7: Supabase and hosted app

- [ ] Import medicines into Supabase `medicines`
- [ ] Import symptom-disease matrix into Supabase tables
- [ ] Import specialist data into `specialist_classifications`
- [ ] Choose final `medicine_ner` source between v1 and v2
- [ ] Keep Python backend as inference source of truth
- [ ] Set `HEALTHMAX_API_URL` for hosted Edge Functions
- [ ] Validate hosted Lovable app against a reachable backend URL

## Phase 8: Voice and WhatsApp

- [ ] Stabilize Twilio WhatsApp flow against the current backend
- [ ] Test voice input path end to end
- [ ] Decide whether ASR remains browser-first or gets a dedicated backend model path
- [ ] Add ASR dataset / training plan only after text triage is stable

## Phase 9: Collaboration work now

### Highest priority

- [ ] Disease ranking quality tuning
- [ ] Benchmark prompt set and regression tests
- [ ] Supabase dataset imports
- [ ] Hosted backend URL integration

### Medium priority

- [ ] Gold NER data curation
- [ ] Specialist classifier
- [ ] Better evaluation reporting

### Lower priority

- [ ] Future hosting plan only after local and hosted validation
- [ ] Whisper fine-tuning
- [ ] Full production ops hardening

## Definition of done for the current milestone

- [x] Local model training works
- [x] Local FastAPI inference works
- [x] Local Lovable app uses the real backend
- [x] Model-to-app path works end to end locally
- [ ] Common benchmark cases reach acceptable quality
- [ ] Hosted Lovable / Supabase path is verified
- [ ] The repo has a clear collaborator-owned task list for the next stage
