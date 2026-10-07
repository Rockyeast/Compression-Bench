"use strict";

// The clean readability test: can the model READ specific records out of the
// compressed table as accurately as out of raw logs? Uses exact ground truth
// (we parse the originals), so there's no LLM-counting noise — only "did it read
// the right fields." Asks for several records in ONE call per form.
//
//   cd ~/ctxfold && npm i openai
//   export OPENAI_API_KEY=sk-...
//   node examples/gpt-lookup-equivalence.js ~/token-bench/ab-openai/fixtures/heavy-logs.json

const fs = require("fs");
const OpenAI = require("openai");
const { compress } = require("../src/index");

const MODEL = process.env.MODEL || "gpt-4o-mini";
const N = 6; // records to look up

function loadLogs(file) {
  let raw = fs.readFileSync(file, "utf8");
  if (file.endsWith(".json")) {
    const fx = JSON.parse(raw);
    const u = fx.messages.find((m) => m.role === "user").content;
    raw = u.split("--- LOGS ---\n")[1] || u;
  }
  return raw;
}

function parseLine(l) {
  const svc = (l.match(/\[([^\]]+)\]/) || [])[1];
  const reqId = (l.match(/reqId=(\d+)/) || [])[1];
  const latency = (l.match(/latency_ms=(\d+)/) || [])[1];
  const status = (l.match(/status=(\d+)/) || [])[1];
  const msg = (l.match(/reqId=\d+ (.*?) latency_ms=/) || [])[1];
  if (svc && reqId && latency && status && msg) return { reqId, service: svc, status, latency_ms: latency, message: msg };
  return null;
}

async function lookup(client, logText, primer, reqIds) {
  const res = await client.chat.completions.create({
    model: MODEL,
    max_completion_tokens: 600,
    messages: [
      { role: "system", content: "You read structured logs and extract exact field values." },
      { role: "user", content:
        `${primer}For EACH of these reqIds: ${reqIds.join(", ")}\n` +
        `find its log line and report service, status, latency_ms, and the message text.\n` +
        `Return ONLY a JSON array of {"reqId","service","status","latency_ms","message"}.\n\n${logText}` },
    ],
  });
  const raw = res.choices[0].message.content.replace(/```json|```/g, "").trim();
  let arr = [];
  try { arr = JSON.parse(raw); } catch { /* leave empty */ }
  return { rows: arr, ptok: res.usage.prompt_tokens };
}

function score(answers, truthByReq) {
  let correct = 0, total = 0;
  const misses = [];
  for (const a of answers) {
    const t = truthByReq[String(a.reqId)];
    if (!t) continue;
    for (const f of ["service", "status", "latency_ms", "message"]) {
      total++;
      if (String(a[f]).trim() === String(t[f]).trim()) correct++;
      else misses.push(`reqId ${a.reqId}.${f}: got "${a[f]}" want "${t[f]}"`);
    }
  }
  return { correct, total, misses };
}

async function main() {
  if (!process.env.OPENAI_API_KEY) { console.error("Set OPENAI_API_KEY"); process.exit(1); }
  const file = process.argv[2] || "/tmp/real.log";
  const raw = loadLogs(file);
  const { text: packed, stats } = compress(raw);
  if (stats.encoder === "none") { console.error("Log encoder didn't fire on this input."); process.exit(1); }

  const parsed = raw.split("\n").map(parseLine).filter(Boolean);
  // pick N distinct random records
  const picks = [];
  const seen = new Set();
  while (picks.length < N && picks.length < parsed.length) {
    const p = parsed[Math.floor(Math.random() * parsed.length)];
    if (seen.has(p.reqId)) continue;
    seen.add(p.reqId); picks.push(p);
  }
  const truthByReq = Object.fromEntries(picks.map((p) => [p.reqId, p]));
  const reqIds = picks.map((p) => p.reqId);

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const primer = "The logs are in a compact columnar format. The 'cols:' line names the columns; " +
    "each row lists those values in order, message last.\n\n";

  const A = await lookup(client, raw, "", reqIds);
  const B = await lookup(client, packed, primer, reqIds);
  const sa = score(A.rows, truthByReq);
  const sb = score(B.rows, truthByReq);

  console.log(`\nmodel: ${MODEL}   records looked up: ${reqIds.length}`);
  console.log(`prompt tokens   raw=${A.ptok}   compressed=${B.ptok}   (${(100 * (1 - B.ptok / A.ptok)).toFixed(1)}% fewer)`);
  console.log(`\nfield accuracy vs ground truth:`);
  console.log(`  RAW         ${sa.correct}/${sa.total}`);
  console.log(`  COMPRESSED  ${sb.correct}/${sb.total}`);
  if (sb.misses.length) { console.log(`\ncompressed misses:`); sb.misses.forEach((m) => console.log("  " + m)); }
  if (sa.misses.length) { console.log(`\nraw misses:`); sa.misses.forEach((m) => console.log("  " + m)); }
  console.log(`\nVERDICT: ${sb.correct >= sa.correct ? "compressed reads as well as (or better than) raw" : "compressed reads worse than raw — adjust format"}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
