"use strict";

// Dictionary-coding experiment. Compresses the same dataset three ways and asks
// gpt-4o-mini to read records out of each, scoring against exact ground truth:
//   1. raw JSON
//   2. ctxfold bare (current default)
//   3. ctxfold + dictionary (experimental)
// Reports tokens AND field accuracy for each, so we see the savings/readability
// tradeoff in one run.
//
//   cd ~/ctxfold && npm i openai && npm i -D gpt-tokenizer
//   export OPENAI_API_KEY=sk-...
//   node experiments/dict-readability.js

const OpenAI = require("openai");
const { compress } = require("../src/index");

let tok; try { const { encode } = require("gpt-tokenizer"); tok = (s) => encode(s).length; }
catch { tok = (s) => Math.ceil(s.length / 4); }

const MODEL = process.env.MODEL || "gpt-4o-mini";
const N = 8;

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
    model: MODEL, max_completion_tokens: 800,
    messages: [
      { role: "system", content: "You read structured product data and extract exact field values." },
      { role: "user", content:
        `${primer}For EACH of these SKUs: ${skus.join(", ")}\n` +
        `report category, price, warehouse, supplier.\n` +
        `Return ONLY a JSON array of {"sku","category","price","warehouse","supplier"}.\n\n${dataText}` },
    ],
  });
  const raw = res.choices[0].message.content.replace(/```json|```/g, "").trim();
  let rows = []; try { rows = JSON.parse(raw); } catch { /* empty */ }
  return { rows, ptok: res.usage.prompt_tokens };
}

function score(answers, truth) {
  let correct = 0, total = 0; const misses = [];
  for (const a of answers) {
    const t = truth[String(a.sku)];
    if (!t) continue;
    for (const f of ["category", "price", "warehouse", "supplier"]) {
      total++;
      if (String(a[f]).trim() === String(t[f]).trim()) correct++;
      else misses.push(`${a.sku}.${f}: got "${a[f]}" want "${t[f]}"`);
    }
  }
  return { correct, total, misses };
}

async function main() {
  if (!process.env.OPENAI_API_KEY) { console.error("Set OPENAI_API_KEY"); process.exit(1); }
  const arr = makeRecords(400);
  const rawText = JSON.stringify(arr);
  const bare = compress(rawText, { countTokens: tok });
  const dict = compress(rawText, { countTokens: tok, dictionary: true });

  const picks = []; const seen = new Set();
  while (picks.length < N) { const p = arr[Math.floor(Math.random() * arr.length)]; if (!seen.has(p.sku)) { seen.add(p.sku); picks.push(p); } }
  const truth = Object.fromEntries(picks.map((p) => [p.sku, p]));
  const skus = picks.map((p) => p.sku);

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const barePrimer = "The data is a compact table: the 'cols:' line names the columns; each row is " +
    "tab-separated values in that order.\n\n";
  const dictPrimer = "The data is a compact table: the 'cols:' line names the columns; each row is " +
    "tab-separated values in that order. Some columns are DICTIONARY-CODED — the legend lists their " +
    "code=value mappings (e.g. category {0=ppe, 1=labels, ...}); translate codes back to the value when answering.\n\n";

  const R = await lookup(client, rawText, "", skus);
  const B = await lookup(client, bare.text, barePrimer, skus);
  const D = await lookup(client, dict.text, dictPrimer, skus);
  const sr = score(R.rows, truth), sb = score(B.rows, truth), sd = score(D.rows, truth);

  const pct = (s) => `${(s.tokenRatio * 100).toFixed(1)}%`;
  console.log(`\nmodel: ${MODEL}   records looked up: ${skus.length}`);
  console.log(`\n                  prompt tokens     savings    field accuracy`);
  console.log(`  raw JSON        ${String(R.ptok).padStart(7)}            —         ${sr.correct}/${sr.total}`);
  console.log(`  ctxfold bare    ${String(B.ptok).padStart(7)}        ${pct(bare.stats).padStart(6)}      ${sb.correct}/${sb.total}`);
  console.log(`  ctxfold +dict   ${String(D.ptok).padStart(7)}        ${pct(dict.stats).padStart(6)}      ${sd.correct}/${sd.total}`);
  if (sd.misses.length) { console.log(`\n+dict misses:`); sd.misses.forEach((m) => console.log("  " + m)); }
  console.log(`\nREAD: if +dict accuracy ≈ bare/raw, dictionary coding is safe to enable. ` +
    `If it drops, the codes cost readability and should stay opt-in.`);
}

main().catch((e) => { console.error(e); process.exit(1); });
