  
**HealthMax**

Bangla AI Health Triage System

Complete Development Master Plan

**Harvard HSIL Hackathon 2026**

One-Month Development Roadmap

Version 1.0  |  March 2026

# **1\. Project Overview**

## **1.1 Vision**

HealthMax is a RAG-powered Bangla AI health triage system that lets community health workers (CHWs) and rural patients describe symptoms in natural Bangla — by text, voice, WhatsApp, or SMS — and receive a structured triage recommendation with probable conditions, urgency level, nearest facility type, and affordable generic medicine options in real time, without requiring internet at the user end.

## **1.2 Core Innovation**

* RAG (Retrieval-Augmented Generation) over real Bangla medical datasets — not hallucination, not a generic chatbot

* Bangla-first: ASR \+ NER \+ classification pipeline built entirely around Bangla language and dialects

* Clinical safety layer — hard rule engine that overrides ML for emergency symptoms

* DGDA drug lookup — returns cheapest generic medicine with real price in BDT

* Zero infrastructure requirement at user end — works on any basic Android or keypad phone

## **1.3 Target Users**

* Primary: Community Health Workers (CHWs) — Bangladesh's 50,000+ frontline DGHS workers

* Secondary: Rural patients directly calling or messaging the system

* Demo audience: HSIL Hackathon 2026 judges

## **1.4 Competitive Advantage Over Previous Winners**

The 2024 winner ZophIA.tech covered mental health in English. HealthMax covers all major diseases in Bangla with clinically validated triage logic, a Bangla dialect normalization pipeline, and a real-world deployment channel (WhatsApp \+ SMS) accessible today.

# **2\. System Architecture**

## **2.1 Full Pipeline**

| USER (CHW / Patient)   │   WhatsApp message / SMS / Browser voice input   ▼ TWILIO WEBHOOK (Free tier)   │   ▼ FASTAPI BACKEND (AWS EC2 t3.small)   │   ├── LAYER 1: Bangla ASR   →  Whisper-medium fine-tuned (HuggingFace)   ├── LAYER 2: NER           →  BanglaBERT fine-tuned on BanglaHealthNER   ├── LAYER 3: RAG Retrieval →  FAISS vector store over disease datasets   ├── LAYER 4: Classifier    →  XGBoost on Symptoms-Disease dataset   ├── LAYER 5: LLM Generator →  GPT-4o API / Amazon Bedrock Claude Haiku   ├── LAYER 6: Triage Rules  →  Hard clinical rule engine (emergency overrides)   └── LAYER 7: Drug Lookup   →  DGDA 50,000-medicine dataset   │   ▼ STRUCTURED RESPONSE (sent back via WhatsApp / SMS / Browser)   ✅ Top 3 Probable Conditions   🚨 Urgency Level: Emergency / Urgent / Self-Care   🏥 Facility: District Hospital / UHC / Community Clinic   💊 Generic Medicine \+ Price (BDT)   📋 CHW Action Instruction in Bangla |
| :---- |

## **2.2 Tech Stack**

| Layer | Tool / Service | Cost | Reason |
| ----- | ----- | ----- | ----- |
| Backend | FastAPI (Python) | Free | Async, fast, perfect for Twilio webhook |
| Server | AWS EC2 t3.small | \~$3 from credit | 2GB RAM — enough for BanglaBERT inference |
| Storage | AWS S3 | \~$0.50 | Dataset & model artifact storage |
| ASR | asif00/whisper-bangla (HuggingFace) | Free | Already fine-tuned on BD dialects |
| Embeddings | paraphrase-multilingual-MiniLM | Free | Multilingual, strong Bangla support |
| Vector Store | FAISS (in-memory) | Free | No external DB needed for hackathon |
| NER Model | sagorsarker/bangla-bert-base | Free | Fine-tune on BanglaHealthNER dataset |
| Classifier | XGBoost | Free | Trains in minutes, interpretable output |
| LLM | GPT-4o API or Amazon Bedrock | \~$5 credit | Natural Bangla response generation |
| TTS | Google Cloud TTS (Bangla WaveNet) | Free tier | Best Bangla voice quality available |
| WhatsApp | Twilio WhatsApp Sandbox | Free | No cost for demo use |
| SMS | Twilio SMS | Free trial | Rural fallback channel |
| CI/CD & IDE | GitHub Codespaces \+ Copilot | Free (Education Pro) | Your IDE, autocomplete, 60hr/mo |
| Training (GPU) | Google Colab (free T4) | Free | Train models here; deploy weights to S3 |
| Monitoring | AWS CloudWatch | Free tier | Basic logging and uptime |
| Domain/SSL | AWS Route 53 \+ ACM | \~$1 from credit | Professional demo URL with HTTPS |

## **2.3 AWS Credit Allocation ($25 Total)**

| Service | Usage | Estimated Cost |
| ----- | ----- | ----- |
| EC2 t3.small (1 month \+ demo days) | Backend server hosting | \~$6.00 |
| Amazon S3 | Datasets \+ model artifacts (\~3GB) | \~$0.10 |
| Amazon Bedrock (Claude Haiku) | LLM fallback if GPT-4o is unavailable | \~$4.00 |
| Amazon Route 53 \+ ACM | Demo domain \+ SSL certificate | \~$1.00 |
| CloudWatch | Monitoring and logs | Free tier |
| Data Transfer | API calls during development | \~$1.00 |
| Buffer / Overruns | Unexpected usage spikes | \~$12.90 |
| TOTAL |  | \~$25.00 |

# **3\. Datasets — Download Before Coding**

All datasets below are free and publicly accessible. Download all Priority 1 and Priority 2 datasets before writing any code — they define the schema of your entire pipeline.

## **3.1 Core Engine Datasets (Priority 1 — Download First)**

| \# | Dataset | Source | Size | Use in HealthMax |
| ----- | ----- | ----- | ----- | ----- |
| 1 | Bangla Symptoms-Disease Dataset | arXiv:2601.12068 \+ Mendeley | 758 relations, 85 diseases, 172 symptoms | Core XGBoost classifier training |
| 2 | Bengali Medical NER Dataset | Kaggle \+ Mendeley: 4tt953xwk2.1 | \~600 statements, 8k words | BanglaBERT NER fine-tuning |
| 3 | Bangla MedER Dataset | Mendeley: BanglaHealthNER | 2,980 annotated medical texts | Symptom entity extraction |
| 4 | Bangla HealthNER Corpus | GitHub | Bengali medical NER corpus | Named entity recognition |

## **3.2 Drug & Medicine Data (Priority 2\)**

| \# | Dataset | Source | Size | Use in HealthMax |
| ----- | ----- | ----- | ----- | ----- |
| 5 | Bangladesh DGDA Medicine Registry | Mendeley: 3x5gsr2jm3.1 | 50,000+ medicines (brand, generic, price) | Drug lookup — cheapest generic \+ BDT price |

## **3.3 ASR & Speech Datasets (Priority 3\)**

| \# | Dataset | Source | Size | Use in HealthMax |
| ----- | ----- | ----- | ----- | ----- |
| 6 | OpenSLR Bangla Speech | openslr.org | 40 hours, 27,308 prompts | Whisper fine-tuning on Bangla speech |
| 7 | Mozilla Common Voice 17 (Bangla) | commonvoice.mozilla.org | 100+ hours | ASR training \+ dialect exposure |

## **3.4 Dialect Datasets (Priority 4\)**

| \# | Dataset | Source | Size | Use in HealthMax |
| ----- | ----- | ----- | ----- | ----- |
| 8 | BanglaDialecto Corpus | GitHub | Noakhali \+ Chittagong regional speech | Dialect normalization pipeline |
| 9 | Bhasha Bichitra Dialect Model | Kaggle | Multi-district dialect data | Regional dialect ASR fine-tuning |

## **3.5 Extended Use Datasets**

| \# | Dataset | Source | Use in HealthMax |
| ----- | ----- | ----- | ----- |
| 10 | B-MHD Mental Health Dataset | scidb.cn | 7,131 Bangla social media texts — mental health triage extension |
| 11 | Bangla LLM Finetune Collection | HuggingFace | Conversation flow fine-tuning for multi-turn dialogue |
| 12 | DiaBD \+ PIMA Diabetes datasets | Already available | Diabetes-specific triage extension (Week 3+) |

# **4\. One-Month Development Roadmap**

The plan is structured into 4 development weeks with a final polish week. Each week builds on the previous. Tasks marked CORE must be completed before moving forward. Tasks marked FLEX can be extended, improved, or skipped if time is short.

## **Week 1 — Foundation & Data Pipeline (Days 1–7)**

| Goal: Working data pipeline \+ trained disease classifier \+ project infrastructure online. |
| :---- |

### **Day 1–2: Environment & Infrastructure Setup \[CORE\]**

* Create GitHub repo with folder structure: /backend, /data, /frontend, /models, /tests, /infra

* Set up AWS EC2 t3.small (Ubuntu 22.04), allocate Elastic IP, open ports 80/443/8000

* Configure GitHub Codespaces with all Python dependencies

* Install: fastapi, uvicorn, transformers, torch, faiss-cpu, openai, twilio, xgboost, sentence-transformers, pandas, scikit-learn, google-cloud-texttospeech

* Download all Priority 1 and Priority 2 datasets to S3 bucket

* Set up Google Colab notebook for GPU-based model training

* Set up Twilio WhatsApp sandbox account (free)

### **Day 3–4: Dataset Processing & FAISS Index \[CORE\]**

* Write data\_processor.py: load, clean, and normalize Symptoms-Disease dataset

* Build symptom vocabulary (172 unique symptoms) and disease label encoder

* Encode all disease-symptom texts using paraphrase-multilingual-MiniLM

* Build and save FAISS index (disease\_rag.index) — this is the RAG backbone

* Process DGDA medicine dataset: index by indication/disease name

* Write unit tests for all data loading functions

### **Day 5–7: XGBoost Disease Classifier \[CORE\]**

* Binarize symptom features using MultiLabelBinarizer

* Train XGBoost classifier (200 trees, max\_depth=6) on 80% split

* Evaluate macro F1 on held-out 20% split — target \> 0.80

* Save trained model as disease\_classifier.json

* Write classifier.py with clean predict() function returning top 3 diseases \+ probabilities

* Write eval\_classifier.py to rerun evaluation on demand

* \[FLEX\] Try LightGBM as alternative — compare F1 scores

## **Week 2 — Core AI Layers (Days 8–14)**

| Goal: NER layer \+ RAG retrieval function \+ clinical rule engine \+ first end-to-end pipeline run. |
| :---- |

### **Day 8–10: BanglaBERT NER Fine-tuning \[CORE\]**

* Load sagorsarker/bangla-bert-base from HuggingFace on Google Colab (T4 GPU)

* Convert BanglaHealthNER \+ MedER datasets to NER token classification format (BIO tagging)

* Labels: B-DISEASE, I-DISEASE, B-SYMPTOM, I-SYMPTOM, B-MEDICINE, I-MEDICINE, O

* Fine-tune for 5–10 epochs; save best checkpoint by validation F1

* Upload fine-tuned model weights to S3; load on EC2 at startup

* \[FLEX\] If GPU time is short, use the pre-fine-tuned bangla-speechprocessing/BanglaNER directly

### **Day 11–12: RAG Retrieval Pipeline \[CORE\]**

* Write rag.py: takes symptom text input, embeds it, queries FAISS index, returns top-5 matching disease records

* Each retrieved record includes: disease name, symptom list, urgency level, specialist type

* Test with 20 manual queries — verify relevant diseases are retrieved

* Tune top\_k parameter (3 vs 5 vs 7\) based on precision

### **Day 13–14: Clinical Rule Engine \[CORE\]**

* Build zarif\_rules.py — hard override rules for emergency and urgent symptom patterns

* Emergency triggers (Bangla): chest pain, difficulty breathing, unconsciousness, seizures, heavy bleeding, stroke signs, high fever in infant

* Urgent triggers: high fever, severe abdominal pain, blood in vomit, severe diarrhea, snakebite

* Rule: if ANY emergency symptom detected → override ML output → return EMERGENCY \+ 999 instruction regardless of model confidence

* Rule: if no emergency flag → pass ML urgency output through unchanged

* IMPORTANT: Rules are binding. ML is advisory. This is the clinical safety net.

* \[CORE\] Run 20 clinical vignettes through the rule engine — 0 unsafe outputs required to proceed

## **Week 3 — Integration & API (Days 15–21)**

| Goal: Full FastAPI backend running end-to-end. Twilio WhatsApp webhook live. First real test with a WhatsApp message. |
| :---- |

### **Day 15–17: FastAPI Backend Integration \[CORE\]**

* Write main.py integrating all layers: ASR → NER → RAG → Classifier → Rules → Drug Lookup → LLM → Response

* Build /webhook/whatsapp POST endpoint for Twilio

* Build /api/triage POST endpoint for browser demo

* Build /health GET endpoint (uptime check)

* Handle errors gracefully: if ASR fails, prompt user to type; if LLM times out, fall back to template response

* Deploy to EC2 with uvicorn, configure nginx reverse proxy, enable HTTPS via AWS ACM

### **Day 16–17: ASR Integration \[CORE\]**

* Write asr.py: load asif00/whisper-bangla model, transcribe audio bytes to Bangla text

* Add confidence check: if avg\_logprob \< threshold, return dialect fallback prompt in Bangla

* Dialect fallback: ask user to speak slowly, or switch to structured yes/no questions

* \[FLEX\] Add dialect identification classifier using BanglaDialecto corpus

### **Day 18–19: LLM Response Generation \[CORE\]**

* Write generator.py: takes extracted symptoms \+ retrieved diseases \+ triage result \+ drug info → GPT-4o prompt

* Prompt in Bangla: specifies output format (probable conditions, urgency, facility, medicine, disclaimer)

* Output must always end with disclaimer: "এটি পরামর্শ, ডাক্তারের বিকল্প নয়"

* Test with Amazon Bedrock Claude Haiku as fallback (uses AWS credit)

* \[FLEX\] Fine-tune a lightweight Bangla LLM for offline capability using Bangla LLM Finetune Collection

### **Day 20–21: DGDA Drug Lookup \[CORE\]**

* Write dgda\_lookup.py: query DGDA dataset by disease/indication → return top 3 cheapest generics

* Output format: generic name, brand example, price in BDT per tablet/unit

* \[FLEX\] Add medicine affordability flag: mark items below ৳5 per unit as "সাশ্রয়ী"

* \[FLEX\] Add drug interaction check for top 10 common combinations

## **Week 4 — Frontend, Testing & Polish (Days 22–28)**

| Goal: Clean demo UI live. 50 clinical vignettes tested. Metrics computed. Pitch materials ready. |
| :---- |

### **Day 22–24: Browser Demo UI \[CORE\]**

* Build index.html with single-page interface — clean, professional, bilingual (Bangla \+ English labels)

* Microphone button using MediaRecorder API — records voice, sends audio blob to /api/triage

* Text input fallback — user can type symptoms in Bangla directly

* Live transcript display: shows ASR output as it processes

* NER visualization: highlight symptom entities in colored tags (symptom \= blue, disease \= red, medicine \= green)

* Output card showing: disease probability bars (Chart.js), urgency badge (color-coded), facility card, drug card

* Use GitHub Copilot to generate boilerplate from descriptive comments — saves 40% of frontend time

* \[FLEX\] Add Bangla TTS playback of the response for audio output

### **Day 25–26: Systematic Testing \[CORE\]**

* Unit Test Phase: input all 172 symptoms individually → verify correct disease mapping → compute macro F1 per disease class

* Clinical Vignette Phase: create 50 real patient-like scenarios from AFMC case experience → run each through full pipeline → classify as Pass / Suboptimal / Unsafe

* Target: 0 unsafe outputs, fewer than 5 suboptimal outputs from 50 vignettes

* Dialect Test Phase: collect or record 10 sentences per dialect (Dhaka standard, Chittagong, Sylheti, Noakhali) → measure WER per dialect

* Usability Test: ask 3 non-technical people to use the browser demo → note confusion points → fix top 3 issues

* Document all results: F1 score, unsafe output count, response latency (target \< 3 seconds), dialect WER

### **Day 27–28: Pitch Preparation \[CORE\]**

* Create pitch slide deck (10 slides max): problem stat → solution → live demo → architecture → datasets → accuracy metrics → deployment model → business case → team → ask

* Key metrics slide must show: number of diseases covered, macro F1 score, 0 unsafe outputs from 50 vignettes, response latency

* Rehearse 3-minute pitch 3 times as full team

* Prepare live demo script — test it on a mobile phone, not just a laptop

* \[FLEX\] Prepare 2-minute extended version in case judges ask for more

## **Buffer Week (Days 29–30) — Contingency & Hardening**

| This week is intentionally left flexible. Use it for whatever slipped in earlier weeks, or for adding one high-impact FLEX feature that judges will find memorable. |
| :---- |

* Fix any test failures from Week 4 validation

* Improve dialect handling if WER is above 30% for Chittagong/Noakhali

* Add mental health triage extension (B-MHD dataset) if time permits

* Stress test the EC2 server — simulate 10 concurrent WhatsApp messages

* Final demo run on mobile phone via WhatsApp — confirm it works exactly as shown in pitch

* Push final code to GitHub, clean README with setup instructions, architecture diagram

# **5\. Repository File Structure**

| healthmax/ ├── backend/ │   ├── main.py              ← FastAPI app \+ Twilio webhook endpoints │   ├── asr.py               ← Whisper ASR layer \+ dialect confidence check │   ├── ner.py               ← BanglaBERT NER symptom extraction │   ├── rag.py               ← FAISS RAG retrieval pipeline │   ├── classifier.py        ← XGBoost disease classifier │   ├── rules.py             ← Clinical rule engine (emergency overrides) │   ├── dgda\_lookup.py       ← Drug lookup from DGDA dataset │   ├── generator.py         ← LLM response generation (GPT-4o / Bedrock) │   └── tts.py               ← Google Cloud TTS Bangla output ├── data/ │   ├── process\_datasets.py  ← Data cleaning \+ FAISS index builder │   ├── raw/                 ← All downloaded datasets (gitignored) │   └── README.md            ← Dataset download instructions \+ sources ├── frontend/ │   ├── index.html           ← Demo UI (voice \+ text input) │   ├── demo.js              ← MediaRecorder \+ API call logic │   └── style.css            ← Clean bilingual styling ├── models/ │   ├── disease\_classifier.json  ← Trained XGBoost model │   └── disease\_rag.index        ← FAISS vector index ├── tests/ │   ├── clinical\_vignettes.csv   ← 50 test scenarios │   ├── eval\_classifier.py       ← Automated F1 \+ accuracy evaluation │   └── dialect\_test\_audio/      ← 10 sentences per dialect for WER testing ├── infra/ │   ├── deploy\_ec2.sh            ← One-command AWS deployment script │   └── nginx.conf               ← Reverse proxy config for HTTPS ├── notebooks/ │   └── banglabert\_finetune.ipynb ← Google Colab NER training notebook ├── requirements.txt └── README.md                    ← Setup guide \+ architecture diagram |
| :---- |

# **6\. Testing Protocol**

## **6.1 Phase 1 — Unit Testing (Automated)**

* Input all 172 symptoms individually as text → verify correct disease mapping against gold standard labels

* Target metric: Macro F1 \> 0.80 across 85 disease classes

* Per-class F1 report printed to console via sklearn classification\_report

* Fail condition: any single high-prevalence disease below F1 \= 0.60

## **6.2 Phase 2 — Clinical Vignette Testing (Manual)**

50 real patient-like scenarios written based on actual clinical cases. Each scenario is run through the full pipeline and classified:

* ✅ PASS: clinically appropriate triage recommendation

* ⚠️ SUBOPTIMAL: correct urgency but imprecise disease or facility suggestion

* ❌ UNSAFE: dangerous recommendation (wrong urgency for emergency, harmful drug suggestion, etc.)

| Scenario | Input Symptoms (Bangla) | Expected Disease | Expected Urgency | System Output | Result |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 1 | বুকে ব্যথা, শ্বাস কষ্ট | Cardiac event | EMERGENCY | (run pipeline) |  |
| 2 | তিন দিন জ্বর, মাথাব্যথা, চোখ লাল, গা ব্যথা | Dengue/Typhoid | URGENT | (run pipeline) |  |
| 3 | পেটব্যথা, বমি, পাতলা পায়খানা | Gastroenteritis | URGENT | (run pipeline) |  |
| 4 | হাত পা কাঁপছে, ঘাম হচ্ছে, মাথা ঘুরছে | Hypoglycemia | URGENT | (run pipeline) |  |
| 5 | গলা ব্যথা, সর্দি, হালকা জ্বর | Upper Respiratory | SELF-CARE | (run pipeline) |  |
| ... | (continue for 50 scenarios) | ... | ... | ... |  |

Target: 0 UNSAFE outputs. Fewer than 5 SUBOPTIMAL outputs. All 50 must be reviewed before the pitch.

## **6.3 Phase 3 — Dialect Testing**

| Dialect | Test Sentences | Target WER | Fallback Strategy |
| ----- | ----- | ----- | ----- |
| Dhaka Standard | 10 | \< 15% | None needed |
| Sylheti | 10 | \< 25% | Dialect normalization layer |
| Barishal | 10 | \< 25% | Dialect normalization layer |
| Noakhali | 10 | \< 35% | Structured yes/no fallback prompting |
| Chittagong | 10 | \< 40% | Structured yes/no fallback prompting |

## **6.4 Phase 4 — Usability Testing**

* 5-person SUS (System Usability Scale) test on the browser demo interface

* Ask 3 non-technical people to interact with the simulator; observe confusion points

* Target: average SUS score \> 70 (Good), no critical blocking confusion

* Fix top 3 usability issues before pitch

## **6.5 Performance Targets Summary**

| Metric | Target | Fail Condition |
| ----- | ----- | ----- |
| Disease classifier Macro F1 | \> 0.80 | Below 0.70 |
| Unsafe clinical outputs from 50 vignettes | 0 | Any single UNSAFE output |
| End-to-end response latency (text input) | \< 3 seconds | Above 5 seconds |
| Dialect WER — Dhaka standard | \< 15% | Above 25% |
| Dialect WER — Chittagong (hardest) | \< 40% | Above 60% |
| SUS usability score | \> 70 / 100 | Below 60 |

# **7\. Risks & Mitigation Strategies**

| \# | Risk | Severity | Mitigation |
| ----- | ----- | ----- | ----- |
| 1 | ASR accuracy on medical terms | HIGH | Fine-tune Whisper on medical vocabulary; add medical-term dictionary boost post-ASR |
| 2 | Dialect divergence (Chittagong, Noakhali) | HIGH | BanglaDialecto normalization pipeline; structured yes/no questions as fallback |
| 3 | Clinically unsafe LLM outputs / hallucinations | HIGH | Rule-based override layer is the safety net — ML is advisory, rules are binding |
| 4 | Low training data for rare diseases | MEDIUM | Focus on top 30 high-prevalence diseases; flag rare symptoms as "see a doctor" |
| 5 | EC2 server goes down during demo | MEDIUM | Pre-record a video backup demo; have a static screenshot fallback slide |
| 6 | User trust in AI medical advice | MEDIUM | System always appends: "এটি পরামর্শ, ডাক্তারের বিকল্প নয়" — framed as triage, not diagnosis |
| 7 | GPT-4o API cost overrun | MEDIUM | Set hard monthly spending limit; fall back to Amazon Bedrock Claude Haiku from AWS credit |
| 8 | BanglaBERT NER fine-tuning takes too long | LOW | Pre-fine-tuned alternatives exist on HuggingFace; use them if Colab time runs out |
| 9 | No live Twilio line approved before demo | LOW | Browser-based voice recorder \+ Flask backend is functionally identical for judges |
| 10 | Data privacy concern from judges | LOW | System is stateless: no patient data stored, no PII collected, call is ephemeral |

# **8\. Pitch Guide (3 Minutes)**

## **8.1 Pitch Script**

| Segment | Duration | What to Say |
| ----- | ----- | ----- |
| Hook | 10 sec | 50 million rural Bangladeshis cannot access proper healthcare. Not because treatment does not exist — but because they cannot communicate with the system. HealthMax fixes that with a single message. |
| Problem | 20 sec | Half of rural patients consult untrained practitioners. Every digital health tool requires a smartphone, internet, and literacy. Until now. |
| Solution | 30 sec | HealthMax: describe your symptoms in any Bangla dialect — by voice or text via WhatsApp. Get a triage recommendation in 5 seconds. Works on a 300-taka phone. No app. No internet required at the user end. |
| Demo | 60 sec | LIVE: type or speak Bangla symptoms → show ASR transcript → NER entity highlights → disease probability chart → urgency badge → drug card with BDT price. Then show an emergency scenario — watch the system override to EMERGENCY instantly. |
| Evidence | 30 sec | Trained on 85 diseases. 172 symptoms. Validated on 50 clinical vignettes: zero unsafe outputs. Response latency under 3 seconds. DGDA drug lookup: 50,000 medicines with real BDT prices. |
| Impact | 30 sec | Deployable through Bangladesh's 50,000 CHW network today, with zero additional infrastructure. This is healthcare for the last mile. |

## **8.2 Demo Sequence**

1. Open browser demo at healthmax.health (or equivalent URL)

2. Type: "তিন দিন ধরে জ্বর, মাথাব্যথা, চোখ লাল, গা ব্যথা"

3. Show each layer activating visually: transcript → NER highlights → disease bars → triage badge → drug card

4. Expected output: Dengue suspected → URGENT → Upazila Health Complex → Paracetamol ৳2.50/tablet

5. Then enter emergency scenario: "বুকে ব্যথা, শ্বাস নিতে পারছি না" — watch EMERGENCY override activate

6. Show: 999 instruction, district hospital direction, rule override notation

## **8.3 Likely Judge Questions & Answers**

| Question | Answer |
| ----- | ----- |
| How do you handle hallucinations? | We use RAG — the model retrieves from a real annotated dataset, not from parametric memory. And the clinical rule engine hard-overrides any output with emergency or urgent flags for known dangerous symptoms. |
| Is this FDA/DGHS approved? | This is a triage support tool for community health workers, not a clinical diagnosis tool. It is positioned within the existing DGHS CHW workflow, not as a replacement for doctors. |
| What happens if ASR fails on a dialect? | The system detects low confidence and switches to structured yes/no prompting in Bangla, which dramatically reduces dialect impact on accuracy. |
| How do you sustain this post-hackathon? | Per-query cost is under $0.01 at scale. DGHS integration model uses existing CHW infrastructure. Grant pathway through DGDA or WHO digital health fund. |

# **9\. Using GitHub Copilot Effectively**

You have GitHub Copilot Education Pro — use it aggressively. Here is how to get the most out of it for each layer of HealthMax.

## **9.1 Best Practices for This Project**

* Write detailed docstrings in Bangla \+ English BEFORE writing any function body — Copilot generates the implementation from the description

* Write comments describing what a function does step by step — Copilot will fill in each step

* For boilerplate (FastAPI routes, Twilio webhook handlers, XGBoost training loop) — just describe what you want in a comment and let Copilot write it

* For the frontend HTML/JS — write descriptive comments like "// show NER entities as colored tags, red for disease, blue for symptom" and Copilot generates the DOM manipulation code

* Use Copilot Chat to explain unfamiliar code (especially the HuggingFace Trainer API and FAISS indexing)

* Do NOT let Copilot write the clinical rule engine — that logic must be hand-written and medically reviewed

## **9.2 Time Savings by Layer**

| Layer | Estimated Without Copilot | Estimated With Copilot | Savings |
| ----- | ----- | ----- | ----- |
| FastAPI backend boilerplate | 4 hours | 1.5 hours | \~60% |
| XGBoost training loop | 2 hours | 45 minutes | \~60% |
| FAISS indexing code | 3 hours | 1 hour | \~65% |
| Frontend HTML/JS demo UI | 6 hours | 2.5 hours | \~58% |
| Test case generation | 3 hours | 1 hour | \~65% |
| DGDA lookup function | 1.5 hours | 30 minutes | \~65% |

# **10\. Progress Tracker**

Use this checklist to track completion. Update it at the end of each day.

## **Week 1 Checklist**

| Task | Status | Owner | Notes |
| ----- | ----- | ----- | ----- |
| GitHub repo \+ folder structure | \[ \] |  |  |
| AWS EC2 t3.small running | \[ \] |  |  |
| All Priority 1+2 datasets downloaded to S3 | \[ \] |  |  |
| data\_processor.py complete | \[ \] |  |  |
| FAISS index built (disease\_rag.index) | \[ \] |  |  |
| XGBoost classifier trained (Macro F1 \> 0.80) | \[ \] |  | Record F1: \_\_\_\_ |
| Twilio sandbox set up | \[ \] |  |  |

## **Week 2 Checklist**

| Task | Status | Owner | Notes |
| ----- | ----- | ----- | ----- |
| BanglaBERT NER fine-tuned \+ weights on S3 | \[ \] |  |  |
| ner.py extract\_symptoms() working | \[ \] |  |  |
| rag.py retrieve\_diseases() tested on 20 queries | \[ \] |  |  |
| rules.py clinical override — 0 unsafe in 20 vignettes | \[ \] |  |  |

## **Week 3 Checklist**

| Task | Status | Owner | Notes |
| ----- | ----- | ----- | ----- |
| FastAPI backend — all routes deployed to EC2 | \[ \] |  |  |
| Twilio webhook /webhook/whatsapp live | \[ \] |  |  |
| Full pipeline test: WhatsApp → response in \< 3s | \[ \] |  | Latency: \_\_\_\_ms |
| asr.py \+ dialect fallback working | \[ \] |  |  |
| generator.py LLM response tested | \[ \] |  |  |
| dgda\_lookup.py drug recommendations working | \[ \] |  |  |
| HTTPS enabled on demo domain | \[ \] |  |  |

## **Week 4 Checklist**

| Task | Status | Owner | Notes |
| ----- | ----- | ----- | ----- |
| Browser demo UI complete and tested on mobile | \[ \] |  |  |
| 50 clinical vignettes tested — 0 UNSAFE outputs | \[ \] |  | UNSAFE count: \_\_\_\_ |
| Dialect WER measured for all 4 dialects | \[ \] |  |  |
| SUS usability test (5 people, score \> 70\) | \[ \] |  | Score: \_\_\_\_ |
| Pitch deck complete (10 slides) | \[ \] |  |  |
| 3 full pitch rehearsals done | \[ \] |  |  |
| Demo tested on real mobile phone via WhatsApp | \[ \] |  |  |

