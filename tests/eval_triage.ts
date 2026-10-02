/**
 * Urgency benchmark for the in-browser triage engine (the one the /triage page runs).
 *
 * Runs every case in tests/clinical_vignettes.csv through runBrowserTriage() with the
 * exported model artifacts in healthmax-ai-assistant/public, and reports urgency
 * accuracy, emergency recall and under-/over-triage.
 *
 * Run from the app folder (uses its vite-node):
 *   cd healthmax-ai-assistant && npx vite-node ../tests/eval_triage.ts
 * Writes tests/results/triage_benchmark.json.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const appPublic = resolve(here, "../healthmax-ai-assistant/public");

// The engine fetches /model/*.json and /doctors.json; serve them from disk.
globalThis.fetch = (async (url: string) => {
  const body = readFileSync(join(appPublic, String(url).replace(/^\//, "")), "utf-8");
  return { ok: true, json: async () => JSON.parse(body) };
}) as unknown as typeof fetch;

const { runBrowserTriage } = await import("../healthmax-ai-assistant/src/lib/browserTriage");

type Level = "EMERGENCY" | "URGENT" | "SELF-CARE";
const LEVELS: Level[] = ["EMERGENCY", "URGENT", "SELF-CARE"];
const RANK: Record<Level, number> = { "SELF-CARE": 0, URGENT: 1, EMERGENCY: 2 };

function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let row: string[] = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const [header, ...body] = rows;
  return body.filter((r) => r.length === header.length).map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

const cases = parseCsv(readFileSync(join(here, "clinical_vignettes.csv"), "utf-8"));
const confusion: Record<string, Record<string, number>> = Object.fromEntries(
  LEVELS.map((e) => [e, Object.fromEntries(LEVELS.map((p) => [p, 0]))]),
);
const details = [];

for (const c of cases) {
  const result = await runBrowserTriage(c.input_bangla);
  const expected = c.expected_urgency as Level;
  const predicted = String(result.urgency_level).toUpperCase() as Level;
  confusion[expected][predicted] = (confusion[expected][predicted] ?? 0) + 1;
  details.push({
    id: Number(c.scenario_id),
    input: c.input_bangla,
    expected,
    predicted,
    outcome: predicted === expected ? "correct" : RANK[predicted] < RANK[expected] ? "under-triage" : "over-triage",
    top_disease: result.top_diseases?.[0]?.disease ?? null,
    note: c.notes,
  });
}

const n = details.length;
const correct = details.filter((d) => d.outcome === "correct").length;
const under = details.filter((d) => d.outcome === "under-triage").length;
const over = details.filter((d) => d.outcome === "over-triage").length;
const emergencies = details.filter((d) => d.expected === "EMERGENCY");
const emergencyHit = emergencies.filter((d) => d.predicted === "EMERGENCY").length;

const summary = {
  engine: "healthmax-ai-assistant/src/lib/browserTriage.ts",
  cases: n,
  urgency_accuracy: +(correct / n).toFixed(3),
  emergency_recall: +(emergencyHit / emergencies.length).toFixed(3),
  under_triage_rate: +(under / n).toFixed(3),
  over_triage_rate: +(over / n).toFixed(3),
  confusion_expected_by_predicted: confusion,
};

mkdirSync(join(here, "results"), { recursive: true });
writeFileSync(join(here, "results/triage_benchmark.json"), JSON.stringify({ summary, details }, null, 2) + "\n");

console.log(JSON.stringify(summary, null, 2));
for (const d of details.filter((x) => x.outcome !== "correct")) {
  console.log(`#${d.id} ${d.outcome}: expected ${d.expected}, got ${d.predicted} | ${d.input}`);
}
