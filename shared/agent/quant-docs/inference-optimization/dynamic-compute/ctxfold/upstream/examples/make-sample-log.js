"use strict";

// Deterministic sample log generator for gpt-logs-equivalence.js.
// Seeded PRNG -> same file every run, with a known ERROR distribution so the
// harness's aggregate question has a hand-checkable ground truth.
//
//   node examples/make-sample-log.js /tmp/sample.log [lines]
//
// Prints the ground truth (ERROR count per service) after writing the file.

const fs = require("fs");

const OUT = process.argv[2] || "/tmp/sample.log";
const LINES = Number(process.argv[3] || 300);

// Small deterministic LCG so output is identical everywhere.
let seed = 42;
function rnd() {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
}
const pick = (a) => a[Math.floor(rnd() * a.length)];

const services = ["api", "auth", "worker", "db"];
const paths = ["/v1/users", "/v1/items", "/v1/orders", "/health", "/v1/login"];
// Weighted level draw: mostly INFO, some WARN, ERRORs skewed toward one service.
function draw() {
  const svc = pick(services);
  const r = rnd();
  let level = "INFO";
  if (r < 0.06 + (svc === "worker" ? 0.09 : 0)) level = "ERROR";
  else if (r < 0.22) level = "WARN";
  return { svc, level };
}

const t0 = Date.parse("2026-07-18T08:00:00Z");
const errBysvc = {};
const rows = [];
for (let i = 0; i < LINES; i++) {
  const { svc, level } = draw();
  if (level === "ERROR") errBySvcCount(svc);
  const ts = new Date(t0 + i * 61000).toISOString().replace(/\.\d{3}Z$/, "Z");
  const status = level === "ERROR" ? pick(["500", "502", "503"]) : "200";
  rows.push(
    `${ts} ${level} [${svc}] reqId=${40000 + i} handled request ` +
    `path=${pick(paths)} status=${status} ms=${Math.floor(rnd() * 90) + 5}`
  );
}
function errBySvcCount(svc) { errBysvc[svc] = (errBysvc[svc] || 0) + 1; }

fs.writeFileSync(OUT, rows.join("\n") + "\n");

const total = Object.values(errBysvc).reduce((a, b) => a + b, 0);
const top = Object.entries(errBysvc).sort((a, b) => b[1] - a[1]);
console.log(`wrote ${LINES} lines -> ${OUT}`);
console.log(`ground truth: ${total} ERROR lines total`);
top.forEach(([s, n]) => console.log(`  ${s}: ${n}`));
console.log(`top service: ${top[0][0]} (${top[0][1]})`);
