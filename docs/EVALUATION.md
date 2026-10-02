# Evaluation: urgency benchmark

**Run date:** 2026-10-02.
**Engine:** `healthmax-ai-assistant/src/lib/browserTriage.ts`, the engine the web app's `/triage` page runs, at submodule commit `2f619cb`.
**Cases:** the 50 Bangla vignettes in [`tests/clinical_vignettes.csv`](../tests/clinical_vignettes.csv): 17 emergency, 21 urgent, 12 self-care. The team wrote these vignettes and **no clinician reviewed them**.
**Raw output:** [`tests/results/triage_benchmark.json`](../tests/results/triage_benchmark.json).

```powershell
cd healthmax-ai-assistant
npx vite-node ../tests/eval_triage.ts
```

## Results

| Metric | Value |
|---|---|
| Urgency accuracy | **46%** (23 / 50) |
| Emergency recall | **53%** (9 / 17) |
| Under-triage (rated less urgent than it is) | **38%** of cases |
| Over-triage (rated more urgent than it is) | 16% of cases |

Rows are the expected urgency; columns are what the engine predicted.

| Expected ↓ / Predicted → | Emergency | Urgent | Self-care |
|---|---|---|---|
| **Emergency** (17) | 9 | 2 | **6** |
| **Urgent** (21) | 0 | 10 | 11 |
| **Self-care** (12) | 0 | 8 | 4 |

**This is not safe for real use.** Six emergencies were told "self-care", including:
- poisoning (বিষ খেয়েছে)
- a dog bite (কুকুরে কামড়েছে)
- breathing stopped (নিঃশ্বাস বন্ধ হয়ে গেছে)
- throat swelling with breathlessness (গলা ফুলে গেছে, শ্বাস নিতে কষ্ট)
- chest pain with sweating, written with the common spelling ব্যাথা instead of ব্যথা

## Why it fails

1. **The emergency keyword list is narrow and spelling-sensitive.** The rule matches exact strings. A common spelling variant (ব্যাথা) or an unlisted phrasing (বিষ খেয়েছে) falls straight through to the disease model.
2. **The disease model can't recognise emergencies it has no disease for.** Poisoning, bites, drowning and anaphylaxis aren't among the 85 diseases, so the engine finds no match and defaults to SELF-CARE.
3. **There is no SELF-CARE label in the data.** All 85 disease records are rated URGENT or EMERGENCY. When a disease matches, mild problems (a cold, acne, indigestion) come out URGENT. When nothing matches, serious problems come out SELF-CARE. Both errors come from the same gap.
4. **Disease ranking was the focus, but urgency is what matters.** The classifier scores well on its own test rows (macro F1 0.73), but that doesn't carry over to urgency on free-text Bangla.

## What a fix would look like (not done; the project is archived)

- Make a **danger-sign protocol** the primary decision (WHO IMCI and ETAT, DGHS guidelines), with spelling-normalised and fuzzy matching, and have specialists sign off on it.
- Add SELF-CARE and "see a doctor within days" levels to the disease records, reviewed by a clinician.
- Treat **emergency recall ≥ 99%** as a release blocker and run this benchmark in CI.
- Have doctors write and review the vignettes, and expand them to about 300 across specialties.
