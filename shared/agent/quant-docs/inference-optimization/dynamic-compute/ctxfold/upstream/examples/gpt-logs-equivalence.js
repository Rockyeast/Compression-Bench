"use strict";

// Proves the compressed log form is as answerable as the raw form.
// Asks gpt-4o-mini an AGGREGATE question (the case lossy compressors fail) on
// BOTH the raw logs and the ctxfold-compressed logs, then judges equivalence.
//
//   cd ~/ctxfold && npm i openai
//   export OPENAI_API_KEY=sk-...
//   node examples/gpt-logs-equivalence.js /tmp/real.log
//   # or point straight at the fixture (auto-extracts the log block):
//   node examples/gpt-logs-equivalence.js ~/token-bench/ab-openai/fixtures/heavy-logs.json
//
// Cross-provider (any OpenAI-compatible endpoint):
//   OPENAI_API_KEY=$ANTHROPIC_API_KEY OPENAI_BASE_URL=https://api.anthropic.com/v1 \
//     MODEL=claude-haiku-4-5 node examples/gpt-logs-equivalence.js /tmp/real.log
//   OPENAI_API_KEY=$GROQ_API_KEY OPENAI_BASE_URL=https://api.groq.com/openai/v1 \
//     MODEL=llama-3.3-70b-versatile node examples/gpt-logs-equivalence.js /tmp/real.log
//
// SLEEP_MS=61000 pauses between the raw and compressed calls (free-tier TPM
// limits). Trim the log file itself if a single request is still too large.

const fs = require("fs");
const OpenAI = require("openai");
const { compress } = require("../src/index");

const MODEL = process.env.MODEL || "gpt-4o-mini";
const SLEEP_MS = Number(process.env.SLEEP_MS || 0);
const MAX_TOKENS = Number(process.env.MAX_TOKENS || 200);
const QUESTION =
  "Using only the log data, answer concisely: (1) how many lines have level ERROR? " +
  "(2) which single service emits the most ERROR lines, and how many? Give the numbers.";

function loadLogs(file) {
  let raw = fs.readFileSync(file, "utf8");
  if (file.endsWith(".json")) {
    const fx = JSON.parse(raw);
    const u = fx.messages.find((m) => m.role === "user").content;
    raw = u.split("--- LOGS ---\n")[1] || u;
  }
  return raw;
}

async function ask(client, label, logText, primer) {
  const res = await client.chat.completions.create({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      { role: "system", content: "You analyze server logs. Be precise and concise." },
      { role: "user", content: `${QUESTION}\n\n${primer}${logText}` },
    ],
  });
  const text = res.choices[0].message.content
    .replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  return { text, ptok: res.usage?.prompt_tokens ?? null };
}

async function judge(client, a, b) {
  const res = await client.chat.completions.create({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      { role: "system", content:
        "Two answers to the same log question. Do they report the SAME numbers and " +
        "same top service? Reply strict JSON {\"match\":true|false,\"note\":\"...\"}." },
      { role: "user", content: `RAW answer:\n${a}\n\nCOMPRESSED answer:\n${b}` },
    ],
  });
  const raw = res.choices[0].message.content
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/```json|```/g, "").trim();
  try { return JSON.parse(raw); }
  catch {
    const m = raw.match(/\{[\s\S]*\}/);
    if (m) { try { return JSON.parse(m[0]); } catch { /* fall through */ } }
    return { match: null, note: "judge parse failed" };
  }
}

async function main() {
  if (!process.env.OPENAI_API_KEY) { console.error("Set OPENAI_API_KEY"); process.exit(1); }
  const file = process.argv[2] || "/tmp/real.log";
  const raw = loadLogs(file);
  const { text: packed, stats } = compress(raw);

  if (stats.encoder === "none") {
    console.error("Log encoder didn't fire on this input — pass the raw log block, not a wrapper.");
    process.exit(1);
  }

  const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL || undefined,
  });
  const primerCompressed =
    "The logs are in a compact columnar format. The 'cols:' line names the columns; " +
    "each row lists those values in order, with the free-text message last.\n\n";

  const A = await ask(client, "raw", raw, "");
  if (SLEEP_MS) await new Promise((r) => setTimeout(r, SLEEP_MS));
  const B = await ask(client, "compressed", packed, primerCompressed);
  if (SLEEP_MS) await new Promise((r) => setTimeout(r, SLEEP_MS));
  const v = await judge(client, A.text, B.text);

  console.log(`\nmodel: ${MODEL}`);
  if (A.ptok != null && B.ptok != null) {
    console.log(`prompt tokens   raw=${A.ptok}   compressed=${B.ptok}   ` +
      `(${(100 * (1 - B.ptok / A.ptok)).toFixed(1)}% fewer)`);
  } else {
    console.log(`prompt tokens   raw=${A.ptok ?? "n/a"}   compressed=${B.ptok ?? "n/a"}   (provider omitted usage)`);
  }
  console.log(`\n--- RAW answer ---\n${A.text}`);
  console.log(`\n--- COMPRESSED answer ---\n${B.text}`);
  console.log(`\nEQUIVALENT: ${v.match}   ${v.note ? "(" + v.note + ")" : ""}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
