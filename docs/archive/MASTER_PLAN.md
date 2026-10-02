# HealthMax: master plan v1 (2026-10-02), NOT BUILT

> **Archived.** This is the "road not taken": the product plan written just before the project was stopped. See [../POSTMORTEM.md](../POSTMORTEM.md) for why. Kept as a reference for competitive analysis, data sources and regulatory constraints.


Status: **draft, waiting for the team's approval.** Nothing in this plan has been built yet.
Name: HealthMax for now. Bangla name candidates are in §13.
This plan replaces `plan.md` (v2.0, March 2026) wherever the two disagree.

---

## 0. How this plan was made

### 0.1 The engineered master prompt (reuse it in every build session)

> **Role.** You are the HealthMax build lead. You work with an expert panel (§0.3) and check
> every deliverable through it.
> **Product.** HealthMax helps people in Bangladesh decide **how urgent** a health problem is,
> **what kind of care** it needs, and **which nearby place actually offers that care**, with
> honest reviews. It also gives verified doctors the tools to review the AI and to issue
> signed e-prescriptions. Bangla first, voice first, works on any phone.
> **Hard rules.**
> (1) Triage, never diagnosis. Show urgency and the next step first, and never list diseases first to patients.
> (2) The AI never prescribes and never states a dose. Only a BMDC-verified doctor prescribes and signs.
> (3) Every clinical rule has a named specialist reviewer and a version.
> (4) Every fact shown about a facility or doctor names its source and date.
> (5) No paid ranking, ever.
> (6) Patient health data stays in Bangladesh and is never sent to a foreign AI API in identifiable form.
> (7) No scraping against a platform's terms.
> **Process.** Plan, then get approval, then build one deliverable at a time. Each deliverable
> goes through the loop in §0.2, and nothing moves on without the user's sign-off (and a doctor's,
> for clinical content). Log every decision in `docs/LOG.md`.

### 0.2 The loop (used for every deliverable)

1. **Set the bar first.** Write testable acceptance criteria before drafting, for example "emergency recall ≥ 99% on the benchmark" or "every form answer is within its word limit".
2. **Draft.**
3. **Panel pass.** Each relevant role in §0.3 reviews the draft against its checklist and lists its top objections.
4. **Revise.** Fix the objections in order of severity, then score again against step 1. Repeat at most three times, or stop sooner once everything passes.
5. **Human gate.** The user reviews. Clinical content is also reviewed by the named doctor.
6. **Log.** Record what was decided, why, and what was rejected.

For model work, the loop is driven by the benchmark. Change one thing, run the benchmark, and keep the change only if the main metric improves and **no emergency case gets worse**.

### 0.3 The expert panel (the lenses every deliverable passes through)

| Role | What they check |
|---|---|
| Clinical psychologist | Health anxiety, fear, stigma, how trust is built, and how people read reviews |
| General physician / internal medicine | Common presentations, dengue and fever pathways, and over- vs under-triage |
| Cardiologist | Chest pain pathways and whether ECG and cardiac care are reachable in time |
| Neurologist | Stroke signs and routing to a place that can do CT around the clock |
| Paediatrician | WHO IMCI danger signs in children and newborns |
| Obstetrician / gynaecologist | Danger signs in pregnancy, emergency obstetric care, and privacy |
| ENT specialist | Airway emergencies, ear and throat red flags |
| Dentist (BDS) | Dental emergencies such as spreading facial infection |
| Psychiatrist / clinical psychologist | Suicide risk and crisis routing (by law only registered professionals give mental-health advice) |
| Ophthalmologist, dermatologist, orthopaedist, emergency physician | Red flags in their own field |
| Public-health researcher | Evaluation design, ethics approval, bias, and claims the evidence can support |
| Data engineer (sources and scraping) | Legality, licences, freshness, matching duplicates across sources |
| Legal and compliance | Telehealth licensing, data residency, the data protection ordinance (PDPO), e-signatures, defamation |
| Business | Who pays, conflicts of interest, cost to run |
| Ops | Keeping it simple: one engine, one database, one admin panel |
| **Patients** (personas, §0.4) | "Would I understand this, trust it, and act on it at 2 am?" |

### 0.4 Patient personas

| Persona | Situation | What they need from HealthMax |
|---|---|---|
| Rahima, 32, garment worker, Gazipur | Low-end Android, little data, 9-month-old with fever at night | Danger signs in plain Bangla, a nearby place with a children's service that is open now, cost |
| Abdul, 64, Rangpur | Button phone, diabetes and hypertension, son lives in Dhaka | Voice or SMS: "where is the nearest place for X", medicines handled through his doctor |
| Karim, 45, CNG driver | Chest pain and sweating at 1 am | "Go now", plus the nearest facility that can do an ECG and cardiac care **now**, with a call button |
| Nusrat, 24, student, Dhaka | Smartphone, irregular periods, anxious | A female gynaecologist, reviews she can trust, privacy, no frightening disease list |
| Sumon, 38, caregiver | Mother needs an MRI | Which places do MRI, their prices, waiting time, reviews |
| Shapla, 50, visually impaired | Uses a screen reader | Fully accessible, voice in and voice out |
| A community clinic provider (CHCP) | Runs a clinic for about 6,000 people | Quick danger-sign check and a referral to the right higher facility |

### 0.5 Panel critique log (loop iterations on this plan, v0 → v1)

| Lens | Objection to the earlier draft | Change made |
|---|---|---|
| Data engineer + legal | "Scrape all Google and Facebook reviews" breaks both platforms' terms, and Google names reviews explicitly in its no-scraping rules | Three permitted sources only: our own verified reviews, Google's official API with attribution, and the full review feeds of facilities that **claim** their profile and connect their own Google and Facebook accounts (§6.3) |
| Legal | The DGHS National Telehealth Guideline requires health data to stay in Bangladesh and requires independently run apps to be licensed | Rule 6 above. A licensing and hosting gate before launch (§9). Claude and ChatGPT connectors only serve **public** data (§8) |
| Cardiologist + neurologist | "Nearest hospital" is not enough. A stroke needs CT around the clock; chest pain needs an ECG now | Triage now outputs a **required capability**, and the map searches for that capability, filtered by "open now" (§4, the core idea) |
| Psychologist | A list of diseases with percentages frightens people and fuels health anxiety | Show urgency and the next step first. Possible causes appear only behind a tap, worded gently. Private mode for sexual and reproductive health and mental health |
| Psychologist | Reviews skew negative; one angry review dominates | Show the rating distribution, recency and counts. AI summaries must cite the reviews they draw on, and give the facility a right of reply |
| Psychiatrist + law | The AI must not counsel | Crisis signs go to a crisis line and registered professionals. No AI therapy |
| Researcher | Features are being planned before quality can be measured | The benchmark and doctor-review loop come before new clinical features (Phase B) |
| Business | `public/doctors.json` lists invented doctors marked `"sponsored": true` | Remove them. Every listed doctor is real and verified. No paid ranking |
| Physician + legal | The patient-facing medicine suggestions amount to prescribing | Removed from the patient side. Medicine data moves to the doctor's prescription tool |
| Ops | Two triage engines (Python and browser) drift apart | One engine, one rules file, one API (§7) |
| Patient (Abdul) | All of this assumes a smartphone | Phase I adds care-finder by SMS and IVR ("nearest place with X") |
| Patient (Nusrat) | She can't pick a female doctor | Filters for doctor gender and language, plus chamber hours |
| Dentist | Dentists were missing from the doctor list | BDS dentists are BMDC-registered and included |

---

## 1. The honest position (summary of the 2026-10-02 critique)

- **The problem is real.** About 73% of health spending is out of pocket. More than 65% of people go first to village doctors. About 54% of out-of-pocket money goes on medicines. A wrong first decision is costly.
- **Triage alone is weak.** Symptom checkers have put the correct diagnosis first in about 34% of cases and given correct triage advice in about 57% (Semigran et al., BMJ 2015). Free chatbots already do Bangla symptom chat.
- **AmarDoctor is a strong direct competitor** for AI triage. It has a knowledge graph built from 1.4M anonymised clinic visits, Bengali voice input, and reports 81% top-1 diagnosis and 91% specialty precision on 185 vignettes. We cannot out-data them on triage alone.
- **So HealthMax's wedge is different:** triage that ends in an action. "You need an ECG within the hour, and here are the three nearest places that can do one now, with their reviews and prices." That joins urgency, capability, location and trust in one step. In our searches no Bangladeshi product joined these (to be confirmed in Phase A).
- **Business:** consumer-only health AI rarely pays (Babylon went bankrupt in 2023, while Ada became profitable selling to health systems and pharma). Revenue has to come from facilities, partners and government (§10), without ever selling ranking.

## 2. Competitive analysis

| Product | What it does | Strength | Gap HealthMax can fill |
|---|---|---|---|
| **AmarDoctor** | AI voice triage in Bengali, specialty routing, clinician SOAP notes, e-prescriptions | Big data (1.4M visits), published evaluation | No service-level care-finder or reviews. Smartphone-centred |
| **DocTime** | Video consultations around the clock, e-prescriptions | Scale, brand | Telemedicine only; doesn't route to physical services |
| **Sasthya Seba** | Appointments, telemedicine, hospital information, ambulance | Wide service menu | Directory-style; no capability search or AI review summaries |
| **Doctorola** | Appointments and video consultations | Doctor network | Same as above |
| **Praava Health** | Clinics plus telemedicine; 500k+ patients | Owns the care | Covers only its own clinics; not a neutral guide |
| **Arogga / Jeeon** | Online pharmacy / pharmacy digitisation | Funded ($5.7M / $2.5M) | Not care navigation. A possible partner |
| **Healtha, DoctorBangladesh.bd** | Doctor and hospital directories (BMDC numbers, chambers, fees, some reviews) | SEO and breadth | Static listings, no triage, no service availability |
| **Test price sites** (e.g. doctordorkar) | Price lists for tests (ECG, MRI) | Useful price data | No availability, triage or reviews |
| **Shastho Batayon 16263** | Government health hotline, free, around the clock | Free and trusted | Voice only; could be a referral partner |
| **Google Maps** | The default "hospital near me", with reviews | Everyone already has it | Doesn't know which services a place has, and doesn't do triage |
| **ChatGPT / Gemini** (ChatGPT Health since Jan 2026) | Free health chat | Fluent, free | Not local, not accountable, doesn't know facilities |
| Practo (India), Zocdoc (US), Ada (B2B) | Directory, booking and reviews; verified-visit reviews; enterprise triage | Proven models abroad | Show what works: verified reviews, revenue from facilities, triage sold to institutions |

**Our position:** *the neutral guide from "what's wrong?" to "the right place, now".* Gaps to verify in Phase A: whether any Bangladeshi product offers search by service ("who does ECG near me, open now"), and how AmarDoctor and DocTime make money.

**The hardest part is not the AI.** It is keeping service availability (has ECG, open now, price) **fresh**. Every directory in the world struggles with this. Our answer: facilities update their own claimed profiles (§6.2), patients report changes after visits, and every field shows when it was last confirmed.

## 3. Product: five pillars

1. **Triage**: danger-sign protocols, urgency, the care needed, and the **required capability**.
2. **Care-finder map**: hospitals, clinics, community clinics and community health workers. Every service at each place, search by service, open hours, prices.
3. **Trust layer**: reviews in the app and permitted external ratings, with AI summaries per facility, department and doctor.
4. **Doctor network**: verified onboarding across all specialties, doctors reviewing the AI, signed e-prescriptions.
5. **Channels**: web app (PWA) first, then Android, then SMS and IVR; Claude and ChatGPT connectors for the public care-finder.

## 4. Core idea: triage → capability → place

```
Symptoms (voice / text, Bangla)
   → language understanding: structured symptoms, age, sex, duration, pregnancy
   → danger-sign protocol engine (doctor-signed rules)
   → output: urgency (Emergency / Within hours / Within days / Self-care)
             + care type (specialty)
             + required capability (e.g. ECG, CT 24/7, emergency obstetric care, children's ward, dental surgery)
   → care-finder: nearest places WITH that capability, open now, sorted by distance and time
   → each place: services, hours, price (with source and date), reviews and AI summary, call / directions
```

Draft capability routing (**for specialist sign-off, not final**):

| Situation (red flags) | Urgency | Capability to search for | Reviewer |
|---|---|---|---|
| Chest pain with sweating, breathlessness, or pain spreading to the arm or jaw | Emergency | ECG now + emergency department + cardiac care | Cardiologist |
| Face drooping, arm weakness, slurred speech (FAST) | Emergency | CT around the clock + stroke-capable hospital | Neurologist |
| Child: unable to drink or breastfeed, vomits everything, convulsions, lethargic (IMCI general danger signs) | Emergency | Children's emergency / inpatient care | Paediatrician |
| Pregnancy: bleeding, severe headache with blurred vision, convulsions, high fever | Emergency | Comprehensive emergency obstetric care | Obstetrician |
| Fever with dengue warning signs (DGHS guideline) | Within hours | Platelet / CBC test + admission | Physician |
| Swelling of the face or neck from a tooth infection, trouble swallowing | Emergency | Emergency department + oral and maxillofacial surgery | Dentist |
| Sudden loss of vision | Emergency | Eye emergency service | Ophthalmologist |
| Thoughts of self-harm | Crisis | Crisis line + psychiatry | Psychiatrist |

## 5. Clinical safety and model plan (replaces the current model work)

**What changes**
- The danger-sign **protocol engine** becomes the core. Rules live in one versioned file, each signed off by a named specialist and built from WHO IMCI, WHO ETAT and DGHS national guidelines.
- **Language understanding**: a model turns Bangla speech or text into structured fields and asks follow-up questions only for what's missing. It never decides urgency on its own.
- The 757-row XGBoost classifier is **moved out of the decision path**. At most it suggests "possible causes" behind a tap, once the benchmark shows it helps.
- Patient-facing medicine suggestions are **removed**.
- **One engine**: the Python and browser engines merge into one service with one rules file.

**Benchmark first (acceptance criteria)**
- About 300 Bangla vignettes across 18+ specialties, written and reviewed by doctors (AmarDoctor used 185). They include tricky everyday phrasings and dialect.
- Metrics:
  - **emergency recall ≥ 99%** (a release blocker);
  - under-triage rate (the dangerous error) and over-triage rate (the costly error), reported separately;
  - specialty routing accuracy;
  - capability routing accuracy.
- The suite runs on every change. Results are logged with a date and the engine version.

**Doctor review loop (also our data moat)**
- Doctors review a sample of anonymised triage sessions, plus every session the engine flagged as uncertain. They mark the correct urgency, specialty and capability.
- Their labels feed the benchmark and protocol updates. Over time this becomes a **doctor-labelled Bangla triage dataset** that competitors can't copy.

**Datasets and sources to research (Phase A)**
- Clinical protocols: WHO IMCI, WHO ETAT, DGHS national guidelines (dengue and others), BMDC guidelines. Check the licence of each before reusing any text.
- Bangla medical NLP and ASR corpora: survey published datasets and their licences. Assess Bangla speech models for medical words.
- Our existing data: trace where `Symptoms.csv` came from. Recover the Excel files misnamed as `.csv` in `src/data`. Fix `tests/clinical_vignettes.csv` (unquoted commas break the columns).

## 6. Care-finder and trust layer

### 6.1 Data model (simple)

`facility` (type, ownership, location, hours, contact, source, last confirmed) → `department` → `service` (from a capability list: ECG, echo, CT, MRI, X-ray, ultrasound, dialysis, ICU, NICU, emergency obstetric care, blood bank, dental surgery…) with price, hours and last confirmed → `doctor` (BMDC number, specialty, gender, languages) ↔ `chamber` (facility, schedule, fee).
Every field carries `source` and `confirmed_at`.

### 6.2 Data sources

| Source | Gives | Reliability | Terms | Use |
|---|---|---|---|---|
| DGHS Facility Registry (FRED API) | 39,428 facility records: names in Bangla and English, type, admin area, public/private | Official, but some records have no coordinates | Public API | Base layer. Sync nightly |
| healthsites.io / OpenStreetMap (HDX) | Facility name, type, activities, coordinates | Community-maintained, uneven | ODbL (attribution, share-alike) | Fill in coordinates, cross-check |
| DGHS HRIS facility profiles | Profile detail for public facilities | Official | Public pages | Departments and beds, if available (verify) |
| Facility self-service (claimed profile) | Services, hours, prices, doctors | Best for freshness, but self-interested | Our terms | Main source of services. Spot-checked by phone |
| Published price lists (e.g. the NINS MRI/CT list) | Prices | Good on the date published | Facts; cite the source | Prices, with date |
| Patient reports after a visit | "ECG wasn't available", "the price was X" | Noisy, but fresh | Our terms | Freshness signal that triggers a re-check |
| BMDC register (verify.bmdc.org.bd) | Is a doctor registered | Authoritative | Captcha, so **manual check per doctor, no scraping** | Doctor verification |
| Google Places API | Rating, rating count, up to 5 reviews, hours, phone | Good | Paid; attribution and link back required; caching limits; Places content shown on a map must use a Google Map (check) | Rating badge on the detail page, with attribution |
| Google Business Profile API (owner OAuth) | **All** reviews of a claimed facility | Good | Owner grants access | Feeds AI summaries for claimed facilities |
| Facebook Graph API (page token from the owner) | Page recommendations (Facebook no longer uses stars) | Fair | Owner grants access | The same, for claimed pages |
| Third-party scrapers (Apify, Outscraper…) | Everything | — | **Against Google's and Meta's terms** | **Not used** |
| Other directories (DoctorBangladesh, Healtha…) | Doctor and chamber listings | Mixed | Their database and terms | **Not copied.** Possible partnership |

Matching duplicates across DGHS, OSM and Google uses Bangla and English name similarity plus distance, with a human review queue for uncertain matches.

### 6.3 Reviews

- **Reviewable entities:** facility, branch/location, department, doctor.
- **Review dimensions:** waiting time, cost transparency, cleanliness, staff behaviour, doctor's communication, overall. No star rating for clinical "outcomes".
- **"Verified visit" badge** when the review is linked to a booking or prescription.
- **Moderation:** health details are removed automatically, abuse is filtered, and borderline reviews go to a human queue. The facility or doctor has a right of reply. A clear policy covers defamation risk.
- **AI summaries** per facility, department and doctor:
  - every claim links to the reviews behind it;
  - the summary shows how many reviews it covers and their date range;
  - it is labelled as AI-written and regenerated weekly;
  - it is built only from reviews we are allowed to use (in-app reviews, and owner-connected Google and Facebook reviews).
- **Display:** rating distribution and recency, not just an average. HealthMax reviews and Google's rating are shown separately and labelled.

## 7. Architecture (simple to run)

- **One database:** Postgres with PostGIS for "nearest with capability" queries. Prototype on Supabase with public and synthetic data only. **Before any real patient data, move to hosting in Bangladesh** (telehealth guideline).
- **One API service** (FastAPI): triage engine, care-finder search, reviews, doctor tools. The web app calls it, and the duplicate engine in the browser is retired.
- **Web app:** the existing React app, redesigned with the brand kit. Map built with MapLibre on OpenStreetMap tiles (free). The Google rating appears only on the facility detail page with attribution.
- **AI models:**
  - patient-facing language understanding runs on a model hosted in Bangladesh, or on **de-identified** text only;
  - review summaries (public text) can use any provider.
- **Scheduled jobs:** nightly DGHS sync, weekly review summaries, monthly check on stale data.
- **One admin panel** with four queues: doctor verification, facility claims, review moderation, flagged triage cases.

## 8. Claude and ChatGPT connectors

- One **remote MCP server**, read-only and public data only. It serves Claude (Connectors Directory: tool titles, `readOnlyHint`) and ChatGPT (Apps SDK, built on MCP).
- Tools for v1:
  - `find_facilities(capability, location, open_now)`
  - `get_facility(id)` (services, hours, prices, sources)
  - `find_doctors(specialty, area, gender)`
  - `get_review_summary(entity)`
  - `list_capabilities()`
- **Not in v1:** triage through connectors. Patient symptoms would flow to a foreign platform, and the engine isn't validated yet. Reconsider after the pilot.
- A Claude **skill or plugin** for internal ops: "verify this doctor", "update facility services from this price list" (with human confirmation).

## 9. Doctor network, e-prescriptions, legal gates

**Onboarding (before launch)**
- Target specialties:
  - general medicine, paediatrics, obstetrics and gynaecology, cardiology, neurology;
  - ENT, dentistry, ophthalmology, dermatology, orthopaedics, psychiatry and clinical psychology;
  - gastroenterology, nephrology, chest medicine, endocrinology, oncology, urology, general surgery, emergency medicine;
  - physiotherapy and nutrition (graduate nutritionists only, per the guideline).
- Verification: BMDC number checked by hand on the official register, plus national ID, a photo and specialty degree certificates.
- The doctor's profile links to chambers (facility, schedule, fee).
- Roles for doctors:
  1. **Protocol owner** for a specialty (signs off its rules);
  2. **Case reviewer** (labels flagged triage cases; paid per review or by honorarium);
  3. **Prescriber**.

**E-prescription**
- BMDC's 2020 telemedicine guidelines and the ICT Act 2006 allow a BMDC-registered doctor to issue an e-prescription.
- The AI may draft a case summary for the doctor. **Only the doctor writes the prescription**, from the medicine database (DGDA list), and signs it.
- **Signature:**
  - v1: a PDF signed in the app by the verified doctor, with a QR verification link and a tamper-evident hash;
  - v2: a PKI digital signature from a certifying authority licensed by the CCA, the legally strongest form under the ICT Act.
- Check which drug categories BMDC rules exclude from telemedicine prescribing.

**Gates before launch (lawyer review required)**
1. Telehealth licence, or joining the national telemedicine marketplace (the guideline requires one or the other for independent apps).
2. Data hosted in Bangladesh. Consent flows that meet PDPO 2025, where health data is sensitive and needs explicit consent.
3. The minimum modules the guideline requires for apps: authentication of doctors and patients, a prescription engine, a health record, and a personal health record.
4. Review policy and defamation counsel.
5. Ethics approval (e.g. BMRC) before any pilot that collects patient data.

## 10. Business model (no paid ranking)

| Stream | Who pays | Conflict check |
|---|---|---|
| Facility "manage your profile" plan (update services, reply to reviews, booking inbox, analytics) | Hospitals, clinics, diagnostic centres | Safe as long as it never buys ranking. State this publicly |
| Booking or referral fee on bookings made | Facility or telemedicine partner | Disclose it. Ranking stays blind to fees |
| Decision support for community clinics, NGOs and health workers | Government, donors, NGOs | Good fit for the mission. Slow sales |
| Insurer or employer navigation | Per member | Small market today |
| Aggregated, anonymised service-gap reports (e.g. "no CT within 50 km") | Researchers, planners | Only with consent and anonymisation under the PDPO |

Unit cost is low. The real costs are verification, moderation, doctor time and compliance. Plan for 3–5 years to sustainability.

## 11. Brand kit plan (Phase C)

- **Psychology brief:** calm, trustworthy, warm, never alarming. Keep red, amber and green **reserved for urgency** and never use them decoratively. The brand's main colour is not red.
- **Deliverables:**
  - a palette of 4–5 brand colours plus a separate urgency scale, all checked for contrast (AA) in light and dark modes;
  - typography: Bangla candidates Noto Sans Bengali, Hind Siliguri, Anek Bangla, Tiro Bangla; Latin to match. Bangla text set slightly larger, with line height of 1.6 or more;
  - a pattern;
  - an icon style;
  - UX rules: one question per screen, voice first, large tap targets, plain "what to do now" wording, sources always visible, private mode;
  - design tokens (CSS and JSON);
  - a guideline page.
- **Three directions to choose from:**
  1. **Kantha** (mending as healing: running-stitch lines; warm and communal);
  2. **Clear water** (calm teal and ink; clinical but gentle);
  3. **Dawn path** (a route-to-care motif; hopeful).
- The loop: each direction is checked by the psychologist and against the personas, then the user chooses.

## 12. Story and pitch material (Phase H)

- **Form-style answers** (problem, target group, solution, role of AI, impact, feasibility, differentiation) in the Urbora structure and word limits, as source material for the pitch deck.
- **Drawn storyboard** (SVG sketch panels, as in Urbora's `storyboards-v2.html`). Working story: Karim's chest pain at 1 am → "Go now" → the nearest place with ECG open now → arrives in time. A second strand: Rahima's baby with fever. Every number on screen comes from the real system.
- **Deck outline** (12 slides) and an optional video built with the Urbora pipeline.

## 13. Name candidates (for later)

Leaning towards দিশা (direction) and ঠিকানা (the right address).
Also considered: পরখ (to check), সময়মতো (in time).
Rejected: আশ্বাস (reads as "don't worry", risky in an emergency), নাড়ি (sounds like নারী), আরোগ্য (Arogga already uses it).
Collision checks pending.

## 14. Roadmap and gates

| Phase | Output | Depends on | Gate |
|---|---|---|---|
| **A. Research and foundations** | Verified competitor table. Data inventory (download a DGHS sample, count coordinate coverage, check what service fields exist). Legal memo for a lawyer. Survey of datasets and clinical protocols. One-page architecture decision | — | User |
| **B. Clinical safety core** | Benchmark, protocol engine v1, one merged engine. Medicine suggestions and fake doctors removed | A | User + a doctor |
| **C. Brand kit** | Three directions → chosen kit, tokens, guideline page | A (positioning) | User |
| **D. Care-finder MVP** | Map, capability search, facility pages (DGHS + OSM data), facility claim flow | A, C | User |
| **E. Trust layer** | In-app reviews, moderation, Google rating badge, owner-connected feeds, AI summaries | D | User |
| **F. Doctor network** | Onboarding and verification, case-review queue, e-prescription v1 | B, legal memo | User + lawyer |
| **G. Connectors** | MCP server for Claude and ChatGPT (public care-finder) | D | User |
| **H. Pitch material** | Form answers, storyboard drawings, deck outline, (video) | A, C (a first draft can start after A) | User |
| **I. Pilot** | One area (one Dhaka area or one upazila) with ethics approval, SMS/IVR care-finder, measured outcomes | B–F, licensing | User + ethics board |

Kept simple on purpose: one repo, one database, one API, one admin panel, and one rules file for clinical logic.

## 15. Decisions needed from the team

1. Approve this plan, or change the order of phases.
2. Doctors: who are the first 3–5 (we need at least a physician, a paediatrician and an obstetrician to start Phase B)?
3. Pilot area preference (a Dhaka area or a rural upazila).
4. OK to remove the patient-facing medicine suggestions and the invented "sponsored" doctors now?

## Sources

- Semigran et al., BMJ 2015, symptom checkers: https://www.semanticscholar.org/paper/Evaluation-of-symptom-checkers-for-self-diagnosis-Semigran-Linder/f7b90d0542b9f1319815a722c013e4090e5ac645
- AmarDoctor paper (arXiv 2510.24724): https://arxiv.org/html/2510.24724v1
- Babylon Health: https://en.wikipedia.org/wiki/Babylon_Health · Ada Health: https://en.wikipedia.org/wiki/Ada_Health
- Out-of-pocket spending 73%: https://www.bssnews.net/news/200361 · Informal providers: https://pmc.ncbi.nlm.nih.gov/articles/PMC12062692/
- Community clinics: https://pmc.ncbi.nlm.nih.gov/articles/PMC7218291/ · 16263: https://bmrcbd.org/Bulletin/bulletin_html/4603/460313.php
- Bangladesh HealthTech landscape: https://tracxn.com/d/explore/healthtech-startups-in-bangladesh/__aA2UNIfBCLiWMXsaQKZbrLTpltfvnFU7qQb7FFdBmA8
- Doctor directories: https://sasthyaseba.com/ · https://doctorbangladesh.bd/ · https://healtha.io/best-telemedicine-apps-in-bangladesh/
- DGHS Facility Registry and API: https://en.info.shr.dghs.gov.bd/technical-support/facility-registry/
- healthsites.io Bangladesh (HDX): https://data.humdata.org/dataset/bangladesh-healthsites
- BMDC verification: https://prescriply.bd/blog/how-to-verify-doctor-bmdc-registration
- BMDC Telemedicine Guidelines 2020: https://www.bmdc.org.bd/docs/BMDC_Telemedicine_Guidelines_July2020.pdf
- DGHS National Telehealth Guideline (2026 upload): https://objectstorage.ap-dcc-gazipur-1.oraclecloud15.com/n/axvjbnqprylg/b/V2Ministry/o/office-dghs/2026/5/652e4061-8b91-4243-85d2-db03d70c005f.pdf
- Personal Data Protection Ordinance 2025: https://en.prothomalo.com/bangladesh/government/teeopu4dfv
- Google Maps scraping and terms: https://www.lobstr.io/blog/is-scraping-google-maps-legal · Places 5-review limit: https://featurable.com/blog/google-places-more-than-5-reviews
- Claude connector submission: https://claude.com/docs/connectors/building/submission · ChatGPT apps: https://openai.com/index/introducing-apps-in-chatgpt/
- Test prices (example): https://doctordorkar.com/diagnostic-test/latest-ecg-test-price-in-bangladesh · NINS MRI/CT list: https://www.nins.gov.bd/nins/index.php/2017-03-10-03-29-21/mri-ct-scan-price-list
