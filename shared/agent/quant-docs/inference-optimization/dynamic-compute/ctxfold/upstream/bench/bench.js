"use strict";

// Benchmark token reduction. Uses gpt-tokenizer for EXACT counts if installed
// (npm i -D gpt-tokenizer); otherwise falls back to a char/4 estimate.
//
//   node bench/bench.js [file]
// With no file, it generates a synthetic templated log sample.

const fs = require("fs");
const { compress } = require("../src/index");

let countTokens;
let tokenizerName = "estimate (chars/4)";
try {
  const { encode } = require("gpt-tokenizer");
  countTokens = (s) => encode(s).length;
  tokenizerName = "gpt-tokenizer (exact, cl100k)";
} catch {
  countTokens = (s) => Math.ceil(s.length / 4);
}

function makeLogs(n = 1200) {
  const levels = ["INFO", "DEBUG", "WARN", "ERROR"];
  const svcs = ["api-gateway", "auth-svc", "billing", "worker-3", "scheduler"];
  const msgs = ["request completed", "cache miss for key", "retry upstream call", "healthcheck ok", "slow query detected"];
  const r = (m) => Math.floor(Math.random() * m);
  const lines = [];
  for (let i = 0; i < n; i++) {
    lines.push(
      `2026-06-${String(r(28) + 1).padStart(2, "0")}T${String(r(24)).padStart(2, "0")}:` +
      `${String(r(60)).padStart(2, "0")}:${String(r(60)).padStart(2, "0")}.${String(r(999)).padStart(3, "0")}Z ` +
      `${levels[r(5) % 4]} [${svcs[r(5)]}] reqId=${r(9999999)} ${msgs[r(5)]} latency_ms=${r(4000)} status=${[200, 404, 500][r(3)]}`
    );
  }
  return lines.join("\n");
}

const file = process.argv[2];
const input = file ? fs.readFileSync(file, "utf8") : makeLogs();

const { text: out, stats } = compress(input, { countTokens });

const tb = countTokens(input);
const ta = countTokens(out);

console.log(`tokenizer:  ${tokenizerName}`);
console.log(`encoder:    ${stats.encoder}`);
console.log(`lossless:   ${stats.lossless}`);
console.log(`chars:      ${input.length} -> ${out.length}  (${((1 - out.length / input.length) * 100).toFixed(1)}% smaller)`);
console.log(`tokens:     ${tb} -> ${ta}  (${((1 - ta / tb) * 100).toFixed(1)}% fewer)`);
