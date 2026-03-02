# HealthMax <small>(_not Baymax_)</small>


**Bangla AI Health Triage System** — Harvard HSIL Hackathon 2026

> Describe your symptoms in any Bangla dialect — by voice or text — and receive a structured triage recommendation in seconds. Works on a basic Android phone via WhatsApp or SMS. No app required.

---

## Architecture

```
User (WhatsApp / SMS / Browser)
        │
        ▼
Twilio Webhook / REST API
        │
        ▼
FastAPI Backend (AWS EC2 t3.small)
  ├── Layer 1: ASR         → Whisper-Bangla (asif00/whisper-bangla)
  ├── Layer 2: NER         → BanglaBERT fine-tuned on BanglaHealthNER
  ├── Layer 3: RAG         → FAISS + paraphrase-multilingual-MiniLM
  ├── Layer 4: Classifier  → XGBoost (85 diseases, 172 symptoms)
  ├── Layer 5: Rules       → Hard clinical override engine (safety net)
  ├── Layer 6: Drug Lookup → DGDA 50,000-medicine registry
  └── Layer 7: LLM         → GPT-4o / Amazon Bedrock Claude Haiku
        │
        ▼
Structured Bangla Response:
  ✅ Top 3 probable diseases
  🚨 Urgency: EMERGENCY / URGENT / SELF-CARE
  🏥 Facility recommendation
  💊 Generic medicine + BDT price
  ⚠️  এটি পরামর্শ, ডাক্তারের বিকল্প নয়।
```

---

## Repository Structure

```
healthmax/
├── backend/
│   ├── main.py              ← FastAPI app + Twilio webhook
│   ├── asr.py               ← Whisper ASR (Layer 1)
│   ├── ner.py               ← BanglaBERT NER (Layer 2)
│   ├── rag.py               ← FAISS RAG retrieval (Layer 3)
│   ├── classifier.py        ← XGBoost classifier (Layer 4)
│   ├── rules.py             ← Clinical rule engine (Layer 5)
│   ├── dgda_lookup.py       ← DGDA drug lookup (Layer 6)
│   ├── generator.py         ← LLM response generation (Layer 7)
│   └── tts.py               ← Google Cloud TTS (FLEX)
├── data/
│   ├── process_datasets.py  ← Data pipeline + FAISS index builder
│   ├── raw/                 ← Downloaded datasets (gitignored)
│   └── README.md            ← Dataset download instructions
├── frontend/
│   ├── index.html           ← Browser demo UI
│   ├── demo.js              ← MediaRecorder + API call logic
│   └── style.css            ← Bilingual styling
├── models/                  ← Trained artifacts (gitignored; from S3)
├── tests/
│   ├── clinical_vignettes.csv   ← 50 test scenarios
│   └── eval_classifier.py       ← Automated evaluation suite
├── infra/
│   ├── deploy_ec2.sh            ← One-command EC2 deployment
│   └── nginx.conf               ← Reverse proxy config
├── notebooks/
│   └── banglabert_finetune.ipynb ← NER fine-tuning on Google Colab
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## Quick Start (Local Development)

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_ORG/healthmax.git
cd healthmax
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys (OpenAI, Twilio, etc.)
```

### 3. Download datasets and build model artifacts

```bash
# See data/README.md for dataset download links
# Place raw files in data/raw/
python data/process_datasets.py --step all
```

### 4. Run the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the browser demo

Navigate to [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)

---

## Deploy to AWS EC2

```bash
# Update EC2_HOST in infra/deploy_ec2.sh, then:
chmod +x infra/deploy_ec2.sh
./infra/deploy_ec2.sh
```

---

## Testing

```bash
# Run full evaluation suite (classifier F1 + 50 clinical vignettes)
python tests/eval_classifier.py

# Rule engine only (fast — no model required)
python tests/eval_classifier.py --phase 2

# Run the rule engine self-test
python backend/rules.py
```

**Targets:**
| Metric | Target |
|--------|--------|
| Disease classifier Macro F1 | > 0.80 |
| Unsafe vignette outputs | 0 |
| End-to-end latency (text) | < 3 seconds |
| Chittagong dialect WER | < 40% |

---

## Collaborator Guide

| File | Owner | Status |
|------|-------|--------|
| `backend/asr.py` | — | 🔲 TODO: implement `transcribe_audio()` |
| `backend/ner.py` | — | 🔲 TODO: implement `extract_symptoms()` |
| `backend/rag.py` | — | 🔲 TODO: implement `retrieve_diseases()` |
| `backend/classifier.py` | — | 🔲 TODO: implement `predict_diseases()` |
| `backend/rules.py` | — | ✅ Skeleton ready — add more keywords |
| `backend/dgda_lookup.py` | — | 🔲 TODO: implement `lookup_drugs()` |
| `backend/generator.py` | — | 🔲 TODO: wire OpenAI / Bedrock calls |
| `data/process_datasets.py` | — | 🔲 TODO: implement all 5 steps |
| `frontend/demo.js` | — | 🔲 TODO: inline NER entity highlighting |
| `notebooks/banglabert_finetune.ipynb` | — | 🔲 TODO: run on Google Colab |

Every function with a `TODO (collaborator):` block has numbered step-by-step instructions in the docstring. Read the docstring before implementing.

---

## Clinical Safety

The **clinical rule engine** (`backend/rules.py`) is the safety net of the system:

- **Emergency keywords** (e.g., chest pain, loss of consciousness, seizure) → **always** override ML output to `EMERGENCY` + 999 instruction.
- **ML output is advisory.** Rules are binding.
- Do **not** modify emergency rules without medical review.
- Run `python tests/eval_classifier.py --phase 2` after every change to rules.py.

---

## License

MIT License — see LICENSE file.

Built for Harvard HSIL Hackathon 2026. Not yet fully developed.
