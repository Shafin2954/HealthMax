# Post-mortem: why HealthMax was archived

**Status:** archived on 2026-10-02. No further development is planned.

## Timeline

| When | What |
|---|---|
| March 2026 | Built for the **Harvard HSIL Hackathon 2026**: Bangla symptom triage for rural Bangladesh. Lovable scaffold, Supabase backend, FastAPI model service |
| March–April 2026 | Trained the BanglaBERT NER, the XGBoost classifier and the retrieval index. Wired up an in-browser engine so the hosted demo needed no server |
| 2026-10-01 | Revisited the project as a possible competition entry. Fixed an urgency bug (unlisted diseases defaulted to SELF-CARE) |
| 2026-10-01 → 02 | Audit, market research and a full product plan. Decided to stop |

## Why it stopped

### 1. A strong competitor already does the core of it

**AmarDoctor** offers AI voice triage in Bengali, specialty routing, a doctor-side assistant, e-prescriptions and a "nearest hospital" assistant. It has:
- a knowledge graph built from **1.4M anonymised clinic visits**;
- a published evaluation (81% top-1 diagnosis, 91% specialty precision on 185 physician-written vignettes);
- a rollout through the Bangladesh Society of General Physicians.

Every major HealthMax feature already existed in AmarDoctor, DocTime, Sasthya Seba, doctor directories or Google Maps. We didn't know about AmarDoctor while building, which is lesson 1.

### 2. The data set a ceiling we couldn't train past

The core dataset has 757 rows of symptom presence for 85 diseases, about 9 per disease, with no age, sex, duration, severity or prevalence. More tuning moves numbers on the dataset's own rows (macro F1 0.73; the dataset paper reports 98%), but it can't make urgency judgements on real Bangla speech reliable. The [benchmark](EVALUATION.md) shows that: **46% urgency accuracy, and 53% of emergencies caught**.

### 3. Consumer triage on its own is a weak business

Consumer symptom checkers rarely pay for themselves:
- Babylon Health went from a $4.2B valuation to bankruptcy in 2023;
- Ada Health reached profitability by selling to health systems and pharma, not patients;
- Bangladesh's best-funded healthtech startup has raised about $5.7M.

A viable version would need a doctor co-founder, a distribution partner and multi-year funding. We had none of those.

### 4. Regulation that a serious version must meet

| Requirement | What it means |
|---|---|
| **National Telehealth Guideline** (DGHS, 2026) | Health data must be hosted in Bangladesh. Independently run telehealth apps need a licence. Only registered professionals may give clinical or mental-health advice |
| **Personal Data Protection Ordinance 2025** | Health data is sensitive personal data and needs explicit consent |
| **BMDC Telemedicine Guidelines 2020** | E-prescriptions only from BMDC-registered doctors |

All of this is doable, but not for a student side project.

## What went wrong in the build (lessons)

1. **Market research came last.** One afternoon of research in week one would have found AmarDoctor and reshaped the idea.
2. **We optimised disease ranking when the product needed urgency.** People need "how fast, and where?", not a ranked list of diseases with percentages. The emergency rules were an afterthought, as a keyword list, and they are what failed.
3. **There was no benchmark until the end.** The vignette file existed, but a quoting bug made it unparseable, and the evaluation script quietly fell back to mock data. So quality was never really measured.
4. **Two engines drifted apart.** We rewrote the Python pipeline in TypeScript for the demo. Fixes made in one didn't reach the other.
5. **Demo shortcuts became product decisions.** The patient screen suggests medicine brands next to a "Final Diagnosis" badge, and lists invented "Partner" doctors marked as sponsored. Each looked harmless in a demo. Each is the wrong pattern for a health product: it amounts to prescribing, it overclaims, and it is paid ranking.
6. **Brand colour collided with meaning.** The crimson primary was nearly the emergency red, so the app looked alarmed all the time. That's fixed in the [brand kit](../brand/README.md).

## What went right

- A real **Bangla medical NER** fine-tune (F1 0.79 on silver labels) and a working pipeline from text and voice to triage, running both server-side and fully in the browser.
- Bilingual UI, voice input, follow-up questions that re-rank candidates, and Supabase auth, roles and data model, all built in a hackathon timeframe.
- An honest stopping point. The audit, the benchmark and this document are part of the work.

## If we started again

Start from the danger-sign protocol (WHO IMCI and ETAT, DGHS guidelines), signed off by clinicians. Make urgency, and **where to go for the capability you need** ("nearest place that can do an ECG now"), the product. Use the language model only to understand speech and text. Measure emergency recall from the first day, and do the market scan before writing any code.

The full "road not taken" plan, with competitive analysis, data sources, legal gates and business model, is archived in [`archive/MASTER_PLAN.md`](archive/MASTER_PLAN.md).
