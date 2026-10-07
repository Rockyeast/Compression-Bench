"use strict";

// Readability proof for the JSON encoder: can the model read records out of the
// compressed TSV table as accurately as out of raw JSON? Exact ground truth, no
// counting. Self-contained (generates its own top-level array of records).
//
//   cd ~/ctxfold && npm i openai
//   export OPENAI_API_KEY=sk-...
//   node examples/gpt-json-equivalence.js
//
// Cross-provider (any OpenAI-compatible endpoint):
//   OPENAI_API_KEY=$ANTHROPIC_API_KEY OPENAI_BASE_URL=https://api.anthropic.com/v1 \
//     MODEL=claude-haiku-4-5 node examples/gpt-json-equivalence.js
//   OPENAI_API_KEY=$GROQ_API_KEY OPENAI_BASE_URL=https://api.groq.com/openai/v1 \
//     MODEL=llama-3.3-70b-versatile node examples/gpt-json-equivalence.js
//
// RECORDS=180 shrinks the dataset (free-tier TPM limits); SLEEP_MS=61000
// pauses between the raw and compressed calls so they don't share a minute.

const OpenAI = require("openai");
const { compress } = require("../src/index");

const MODEL = process.env.MODEL || "gpt-4o-mini";
const N = 6;
const RECORDS = Number(process.env.RECORDS || 400);
const SLEEP_MS = Number(process.env.SLEEP_MS || 0);
const MAX_TOKENS = Number(process.env.MAX_TOKENS || 700);

function makeRecords(n = 400) {
  const cats = ["packaging", "equipment", "labels", "ppe", "electrical"];
  const whs = ["CHI-1", "DAL-2", "ATL-3", "SEA-4"];
  const sup = ["Acme", "Boruka", "VoltCo", "PrintZ"];
  const r = (m) => Math.floor(Math.random() * m);
  const arr = [];
  for (let i = 0; i < n; i++) {
    arr.push({
      sku: `NW-${1000 + i}`, name: `Item ${i}`, category: cats[r(cats.length)],
      price: Number((Math.random() * 2500 + 5).toFixed(2)), qty: r(10000),
      warehouse: whs[r(whs.length)], supplier: sup[r(sup.length)], hazmat: Math.random() < 0.1,
    });
  }
  return arr;
}

async function lookup(client, dataText, primer, skus) {
  const res = await client.chat.completions.create({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      { role: "system", content: "You read structured product data and extract exact field values." },
      { role: "user", content:
        `${primer}For EACH of these SKUs: ${skus.join(", ")}\n` +
        `report price, qty, warehouse, supplier.\n` +
        `Return ONLY a JSON array of {"sku","price","qty","warehouse","supplier"}.\n\n${dataText}` },
    ],
  });
  const raw = res.choices[0].message.content
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/```json|```/g, "").trim();
  let arr = []; try { arr = JSON.parse(raw); } catch {
    const m = raw.match(/\[[\s\S]*\]/);
    if (m) { try { arr = JSON.parse(m[0]); } catch { /* empty */ } }
  }
  if (!arr.length && process.env.DEBUG) console.error("unparsed reply:\n" + raw.slice(0, 600));
  return { rows: arr, ptok: res.usage?.prompt_tokens ?? null };
}

function score(answers, truth) {
  let correct = 0, total = 0; const misses = [];
  for (const a of answers) {
    const t = truth[String(a.sku)];
    if (!t) continue;
    for (const f of ["price", "qty", "warehouse", "supplier"]) {
      total++;
      if (String(a[f]).trim() === String(t[f]).trim()) correct++;
      else misses.push(`${a.sku}.${f}: got "${a[f]}" want "${t[f]}"`);
    }
  }
  return { correct, total, misses };
}

async function main() {
  if (!process.env.OPENAI_API_KEY) { console.error("Set OPENAI_API_KEY"); process.exit(1); }
  const arr = makeRecords(RECORDS);
  const rawText = JSON.stringify(arr);
  const { text: packed, stats } = compress(rawText);

  const picks = [];
  const seen = new Set();
  while (picks.length < N) {
    const p = arr[Math.floor(Math.random() * arr.length)];
    if (seen.has(p.sku)) continue; seen.add(p.sku); picks.push(p);
  }
  const truth = Object.fromEntries(picks.map((p) => [p.sku, p]));
  const skus = picks.map((p) => p.sku);

  const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL || undefined,
  });
  const primer = "The data is a compact table: the 'cols:' line names the columns; each row is " +
    "tab-separated values in that order, one row per record.\n\n";

  const A = await lookup(client, rawText, "", skus);
  if (SLEEP_MS) await new Promise((r) => setTimeout(r, SLEEP_MS));
  const B = await lookup(client, packed, primer, skus);
  const sa = score(A.rows, truth), sb = score(B.rows, truth);

  console.log(`\nmodel: ${MODEL}   dataset: ${RECORDS} records   probed: ${skus.length}   encoder: ${stats.encoder}`);
  console.log(`chars ${stats.charsBefore}->${stats.charsAfter} (${(stats.charRatio * 100).toFixed(1)}% smaller)`);
  if (A.ptok != null && B.ptok != null) {
    console.log(`prompt tokens   raw=${A.ptok}   compressed=${B.ptok}   (${(100 * (1 - B.ptok / A.ptok)).toFixed(1)}% fewer)`);
  } else {
    console.log(`prompt tokens   raw=${A.ptok ?? "n/a"}   compressed=${B.ptok ?? "n/a"}   (provider omitted usage)`);
  }
  console.log(`\nfield accuracy vs ground truth:`);
  console.log(`  RAW         ${sa.correct}/${sa.total}`);
  console.log(`  COMPRESSED  ${sb.correct}/${sb.total}`);
  if (sb.misses.length) { console.log(`\ncompressed misses:`); sb.misses.forEach((m) => console.log("  " + m)); }
  if (sa.misses.length) { console.log(`\nraw misses:`); sa.misses.forEach((m) => console.log("  " + m)); }
  if (sa.total === 0 && sb.total === 0) {
    console.log(`\nVERDICT: no scoreable answers on either side — parse or alignment failure, not a readability result (rerun with DEBUG=1)`);
  } else {
    console.log(`\nVERDICT: ${sb.correct >= sa.correct ? "compressed reads as well as (or better than) raw" : "compressed reads worse — adjust format"}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
