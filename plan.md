**HealthMax**

Bangla AI Health Triage System

Updated Development Plan

Version 2.0 | March 25, 2026

# 1. Project Status

## 1.1 What Is True Now

HealthMax now has a working **local end-to-end path**:

1. the user opens the Lovable app
2. the Lovable `/triage` page sends a request to the local FastAPI backend
3. the backend runs:
   - Bangla medical NER
   - disease retrieval / RAG
   - XGBoost disease classification
   - clinical safety rules
   - DGDA medicine lookup
4. the Lovable UI renders:
   - probable diseases
   - urgency
   - facility recommendation
   - medicine suggestions

This path has already been browser-verified locally.

## 1.2 What Is No Longer The Main Problem

The main problem is **not** basic wiring anymore.

These are already working locally:

- model training
- backend inference
- Lovable-to-backend integration
- browser rendering of model-backed results

## 1.3 What The Main Problem Is Now

The main problem is now **prediction quality**, especially on ambiguous fever-like cases such as:

- dengue vs malaria
- dengue vs swine flu
- weak-signal symptom sets with overlapping diseases

# 2. Current Architecture

## 2.1 Verified Local Architecture

```text
Lovable App (/triage)
        |
        v
VITE_HEALTHMAX_API_URL
        |
        v
FastAPI /api/triage
        |
        +--> NER (BanglaBERT fine-tuned locally)
        +--> RAG (FAISS + sentence-transformer embeddings)
        +--> Classifier (XGBoost)
        +--> Rules (emergency / urgent override)
        +--> DGDA lookup (medicine.csv)
        |
        v
Structured JSON response
        |
        v
Lovable UI
```

## 2.2 Hosted Architecture Target

This is the target after local quality is stronger:

```text
Lovable Hosted App / Supabase Edge Function
        |
        v
Configurable backend URL
        |
        v
Python FastAPI model backend
```

Important:

- hosted integration is a **next phase**
- AWS is **not** part of the active implementation path right now

# 3. Canonical Data and Models

## 3.1 Canonical Datasets

### Disease classifier and disease retrieval

- `data/raw/Symptoms.csv`

### Medicine lookup

- `assets/medicine.csv`

### Specialist routing candidate

- `healthmax-ai-assistant/src/data/specialist_classification.csv`

### Silver-label NER source data

- `data/raw/Symptoms.csv`
- `healthmax-ai-assistant/src/data/medicine_ner_v2.csv`
- `healthmax-ai-assistant/src/data/medicine_ner.csv`
- `healthmax-ai-assistant/src/data/specialist_classification.csv`

## 3.2 Current Trained Artifacts

### Disease classifier

- `models/disease_classifier.json`
- `models/label_encoder.json`
- `models/symptom_list.json`

Current metric:

- macro F1: `0.7255`

### Disease retrieval / RAG

- `models/disease_rag.index`
- `models/disease_records.json`
- `models/rag_config.json`

Current runtime:

- embeddings on CUDA
- FAISS on CPU

### Bangla medical NER

- `models/ner-banglabert-medical/`
- `models/ner_training_summary.json`

Current metric:

- validation F1: `0.7896`

# 4. Updated Roadmap

## Phase 1. Local Quality Hardening

Goal:

- make local predictions good enough to trust for demo-quality cases

Tasks:

- tune hybrid disease ranking weights
- improve Bangla symptom normalization
- improve disease mention handling
- reduce noisy tied predictions
- create benchmark cases with expected top-3 outputs
- add regression tests for those prompts

Success criteria:

- common benchmark prompts produce clinically sensible top-3 outputs
- emergency prompts still override correctly
- medicine-only prompts do not invent fake disease lists

## Phase 2. Evaluation and Benchmarking

Goal:

- move from ad hoc testing to repeatable evaluation

Tasks:

- create benchmark prompt set in Bangla
- define expected outputs for:
  - chest pain / breathing emergency
  - diarrhea / dehydration
  - measles mention
  - dengue-like fever
  - medicine-only input
- add scripts to rerun backend checks automatically
- document score changes when models are retrained

Success criteria:

- one repeatable local benchmark suite exists
- regressions are easy to catch before hosted testing

## Phase 3. NER Quality Upgrade

Goal:

- move beyond silver-label-only NER

Tasks:

- replace silver labels with reviewed gold BIO annotations
- add label-wise NER evaluation:
  - symptom
  - disease
  - medicine
- improve short disease phrase extraction

Success criteria:

- cleaner entity extraction
- lower noise from imperfect silver labels

## Phase 4. Specialist Routing

Goal:

- add a real specialist recommendation model if it improves results

Tasks:

- train a specialist-routing model from `specialist_classification.csv`
- compare against current lookup-based behavior
- integrate specialist output into final response

Success criteria:

- specialist output is materially better than the current fallback path

## Phase 5. Hosted Supabase / Lovable Integration

Goal:

- move the already-working local backend path into a hosted flow

Tasks:

- import datasets into Supabase tables
- set `HEALTHMAX_API_URL` for Edge Functions
- make Lovable hosted app call the reachable model backend
- validate hosted `/triage` behavior

Success criteria:

- hosted Lovable app returns model-backed predictions through the Python backend

## Phase 6. Voice and Messaging

Goal:

- validate WhatsApp and voice paths only after text triage is stable

Tasks:

- stabilize Twilio WhatsApp flow
- test voice endpoint end to end
- decide whether ASR remains browser-first or gets a dedicated backend training track

Success criteria:

- at least one messaging path and one voice path work against the same backend

# 5. Immediate Priorities

These are the active priorities right now:

1. improve disease ranking quality
2. build benchmark and regression tests
3. import Supabase datasets
4. validate hosted model-backed app flow

# 6. Future Hosting Plan Only

AWS is no longer the active implementation track.

If hosting is needed later, AWS can be revisited as an optional deployment target.

## 6.1 Optional Future Hosting Targets

Possible future options:

- AWS EC2
- any VPS
- container hosting
- local tunnel for short-term hosted testing

## 6.2 If AWS Is Reintroduced Later

Use AWS only after:

- local quality is acceptable
- hosted Lovable path is validated
- Supabase integration is stable

If that happens, AWS would be used for:

- serving FastAPI publicly
- optional HTTPS/reverse proxy
- optional artifact storage

But this is a **future hosting plan only**, not current project scope.

# 7. Risks and Mitigations

## 7.1 Current Real Risks

### Risk 1: weak disease ranking on overlapping symptom sets

Mitigation:

- improve hybrid ranking
- create benchmark prompts
- compare model output before and after each retrain

### Risk 2: silver-label NER ceiling

Mitigation:

- move to reviewed gold BIO labels

### Risk 3: hosted path diverges from local path

Mitigation:

- keep Python backend as inference source of truth
- keep hosted flow thin and configurable

### Risk 4: emergency logic weakened during tuning

Mitigation:

- keep rules binding
- regression test emergency prompts separately

# 8. Definition of Done For The Next Milestone

The next milestone is done when:

- local Lovable app still works with the backend
- benchmark prompts are saved and repeatable
- disease ranking quality is clearly stronger on common cases
- hosted Lovable / Supabase path is validated against the same backend

# 9. Notes For Collaborators

Read these first:

- `PROJECT_SITUATION.md`
- `tasks.md`
- `README.md`

Primary code areas for the next phase:

- `backend/fusion.py`
- `backend/main.py`
- `backend/ner.py`
- `backend/rag.py`
- `data/build_ner_dataset.py`
- `training/train_ner.py`
- `healthmax-ai-assistant/src/pages/Triage.tsx`
- `healthmax-ai-assistant/supabase/functions/healthmax-triage/index.ts`

# 10. Bottom Line

HealthMax has moved from an idea-plus-mockup into a working local prototype.

The plan is now:

- keep local model-to-app working
- improve quality
- validate hosted integration
- treat AWS only as a later hosting option
