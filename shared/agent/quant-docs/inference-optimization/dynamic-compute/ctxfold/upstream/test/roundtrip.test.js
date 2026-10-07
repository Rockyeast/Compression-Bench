"use strict";

const test = require("node:test");
const assert = require("node:assert");
const { compress, decompress } = require("../src/index");

// Build a realistic templated log sample.
function makeLogs(n = 200) {
  const levels = ["INFO", "DEBUG", "WARN", "ERROR"];
  const svcs = ["api-gateway", "auth-svc", "billing", "worker-3"];
  const msgs = ["request completed", "cache miss for key", "retry upstream call", "healthcheck ok"];
  const r = (m) => Math.floor(Math.random() * m);
  const lines = [];
  for (let i = 0; i < n; i++) {
    lines.push(
      `2026-06-${String(r(28) + 1).padStart(2, "0")}T${String(r(24)).padStart(2, "0")}:` +
      `${String(r(60)).padStart(2, "0")}:${String(r(60)).padStart(2, "0")}.${String(r(999)).padStart(3, "0")}Z ` +
      `${levels[r(4)]} [${svcs[r(4)]}] reqId=${r(9999999)} ${msgs[r(4)]} latency_ms=${r(4000)} status=${[200, 404, 500][r(3)]}`
    );
  }
  return lines.join("\n");
}

const samples = {
  "templated logs": makeLogs(200),
  "logs no trailing nl": makeLogs(120),
  "logs with trailing nl": makeLogs(120) + "\n",
  "logs + junk lines":
    makeLogs(60) + "\n!!! panic: weird  spacing and no structure\n\n" + makeLogs(60),
  "tiny": "2026-01-01T00:00:00Z INFO [svc] reqId=1 hi status=200",
  "prose": "The quick brown fox jumps over the lazy dog. ".repeat(30),
  "empty": "",
};

for (const [name, text] of Object.entries(samples)) {
  test(`lossless round-trip: ${name}`, () => {
    const { text: out } = compress(text);
    assert.strictEqual(decompress(out), text, "decompress must reproduce input exactly");
  });
}

test("logs actually compress", () => {
  const { stats } = compress(makeLogs(300));
  assert.strictEqual(stats.encoder, "logs");
  assert.ok(stats.charRatio > 0.2, `expected >20% reduction, got ${(stats.charRatio * 100).toFixed(1)}%`);
});

test("prose passes through untouched", () => {
  const text = "Plain narrative text with no repeated structure at all. ".repeat(20);
  const { text: out, stats } = compress(text);
  assert.strictEqual(stats.encoder, "none");
  assert.strictEqual(out, text);
});

test("never lossy: compressed output always decompresses to input", () => {
  for (let i = 0; i < 25; i++) {
    const text = makeLogs(50 + i * 7);
    const { text: out } = compress(text);
    assert.strictEqual(decompress(out), text);
  }
});

// ---- JSON-array encoder ----------------------------------------------------

function makeRecords(n = 200) {
  const cats = ["packaging", "equipment", "labels", "ppe"];
  const whs = ["CHI-1", "DAL-2", "ATL-3"];
  const r = (m) => Math.floor(Math.random() * m);
  const arr = [];
  for (let i = 0; i < n; i++) {
    arr.push({
      sku: `NW-${1000 + i}`, name: `Item ${i}`, category: cats[r(cats.length)],
      price: Number((Math.random() * 2500 + 5).toFixed(2)), qty: r(10000),
      warehouse: whs[r(whs.length)], hazmat: Math.random() < 0.1,
    });
  }
  return JSON.stringify(arr);
}

test("json: lossless round-trip (uniform array)", () => {
  const text = makeRecords(200);
  const { text: out, stats } = compress(text);
  assert.strictEqual(stats.encoder, "json");
  assert.strictEqual(decompress(out), JSON.stringify(JSON.parse(text)));
});

test("json: actually compresses", () => {
  const { stats } = compress(makeRecords(300));
  assert.ok(stats.charRatio > 0.4, `expected >40% reduction, got ${(stats.charRatio * 100).toFixed(1)}%`);
});

test("json: nested/odd objects kept verbatim, still lossless", () => {
  const arr = JSON.parse(makeRecords(20));
  arr.splice(5, 0, { sku: "NESTED", meta: { a: 1, b: [1, 2, 3] } });
  arr.splice(12, 0, { totally: "different", shape: true });
  const text = JSON.stringify(arr);
  const { text: out } = compress(text);
  assert.strictEqual(decompress(out), JSON.stringify(JSON.parse(text)));
});

test("json: nulls and mixed-type columns round-trip", () => {
  const arr = [];
  for (let i = 0; i < 20; i++) arr.push({ id: i, note: i % 3 ? "ok" : null, score: i % 2 ? i * 1.5 : "n/a" });
  const text = JSON.stringify(arr);
  const { text: out } = compress(text);
  assert.strictEqual(decompress(out), JSON.stringify(JSON.parse(text)));
});

test("json: non-array passes through untouched", () => {
  const text = JSON.stringify({ a: 1, b: 2, c: [1, 2, 3] });
  const { text: out, stats } = compress(text);
  assert.strictEqual(stats.encoder, "none");
  assert.strictEqual(out, text);
});

test("json: wrapped array {results:[...]} round-trips", () => {
  const text = JSON.stringify({ results: JSON.parse(makeRecords(200)) });
  const { text: out, stats } = compress(text);
  assert.strictEqual(stats.encoder, "json");
  assert.strictEqual(decompress(out), JSON.stringify(JSON.parse(text)));
});

test("json: wrapped array with sibling metadata is fully preserved", () => {
  const obj = { count: 200, page: 1, nextCursor: "abc123", data: JSON.parse(makeRecords(200)) };
  const text = JSON.stringify(obj);
  const { text: out } = compress(text);
  assert.strictEqual(decompress(out), JSON.stringify(JSON.parse(text)));
});

test("json: two competing record arrays decline (ambiguous)", () => {
  const text = JSON.stringify({ a: JSON.parse(makeRecords(10)), b: JSON.parse(makeRecords(10)) });
  const { stats } = compress(text);
  assert.strictEqual(stats.encoder, "none");
});

test("json: dictionary coding round-trips losslessly (opt-in)", () => {
  const text = makeRecords(200);
  const { text: out, stats } = compress(text, { dictionary: true });
  assert.strictEqual(stats.encoder, "json");
  assert.strictEqual(decompress(out), JSON.stringify(JSON.parse(text)));
});

test("json: dictionary coding compresses more than bare on low-cardinality cols", () => {
  const text = makeRecords(300);
  const bare = compress(text);
  const dict = compress(text, { dictionary: true });
  assert.ok(dict.stats.charsAfter < bare.stats.charsAfter,
    `dict (${dict.stats.charsAfter}) should beat bare (${bare.stats.charsAfter})`);
});

test("json: dictionary is OFF by default (no behavior change)", () => {
  const text = makeRecords(100);
  assert.strictEqual(compress(text).text, compress(text, { dictionary: false }).text);
});

// ---- validate() ------------------------------------------------------------

const { validate } = require("../src/index");

test("validate: accepts a well-formed payload", () => {
  for (const text of [makeLogs(80), makeRecords(80), makeCsv(80)]) {
    const { text: out } = compress(text);
    const r = validate(out);
    assert.strictEqual(r.valid, true, `expected valid for ${r.encoder}`);
  }
});

test("validate: rejects a non-ctxfold string", () => {
  const r = validate("just some plain text, not folded");
  assert.strictEqual(r.valid, false);
});

test("validate: detects column drift (a dropped cell)", () => {
  const { text: out } = compress(makeRecords(80));
  // Corrupt one row: remove its last tab-separated cell.
  const lines = out.split("\n");
  const rowsIdx = lines.indexOf("rows:") + 1;
  lines[rowsIdx] = lines[rowsIdx].split("\t").slice(0, -1).join("\t");
  const r = validate(lines.join("\n"));
  assert.strictEqual(r.valid, false);
  assert.match(r.reason, /cells|columns/);
});

test("validate: detects truncated rows", () => {
  const { text: out } = compress(makeLogs(80));
  const lines = out.split("\n");
  const r = validate(lines.slice(0, lines.length - 5).join("\n")); // drop 5 rows
  assert.strictEqual(r.valid, false);
});

// ---- CSV / TSV encoder -----------------------------------------------------

function makeCsv(n = 200, delim = ",") {
  const whs = ["CHI-1", "DAL-2", "ATL-3"];
  const r = (m) => Math.floor(Math.random() * m);
  const rows = [["id", "date", "region", "warehouse", "qty"].join(delim)];
  for (let i = 0; i < n; i++) {
    rows.push([`USR-${1000 + i}`, `2026-06-${String(r(28) + 1).padStart(2, "0")}`, "NA", whs[r(whs.length)], r(9999)].join(delim));
  }
  return rows.join("\n");
}

test("csv: lossless round-trip with factorable redundancy", () => {
  const text = makeCsv(250);
  const { text: out, stats } = compress(text);
  assert.strictEqual(stats.encoder, "csv");
  assert.strictEqual(decompress(out), text);
});

test("csv: trailing newline is byte-exact", () => {
  const text = makeCsv(120) + "\n";
  const { text: out } = compress(text);
  assert.strictEqual(decompress(out), text);
});

test("tsv: lossless round-trip", () => {
  const text = makeCsv(150, "\t");
  const { text: out, stats } = compress(text);
  assert.strictEqual(stats.encoder, "csv");
  assert.strictEqual(decompress(out), text);
});

test("csv: ragged rows kept verbatim, still lossless", () => {
  const lines = makeCsv(60).split("\n");
  lines.splice(10, 0, "BROKEN,row,with,too,many,fields,here");
  const text = lines.join("\n");
  const { text: out } = compress(text);
  assert.strictEqual(decompress(out), text);
});

test("csv: quoted fields decline (safety), pass through untouched", () => {
  const text = 'a,b\n1,"hello, world"\n2,"x"\n3,y\n4,z\n5,w';
  const { text: out, stats } = compress(text);
  assert.strictEqual(stats.encoder, "none");
  assert.strictEqual(out, text);
});
