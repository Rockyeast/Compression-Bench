"use strict";

const test = require("node:test");
const assert = require("node:assert");
const { compress } = require("../src/index");
const { profile, renderProfile } = require("../src/profile");

// ---------------------------------------------------------------------------
// Fixtures (deterministic — no randomness in tests).

function jsonFixture(n = 40) {
  const cats = ["packaging", "equipment", "labels", "ppe", "electrical"];
  const arr = [];
  for (let i = 0; i < n; i++) {
    arr.push({
      sku: `NW-${1000 + i}`, name: `Item ${i}`, category: cats[i % cats.length],
      price: Number((i * 3.17 + 5).toFixed(2)), qty: (i * 37) % 10000,
      warehouse: `WH-${["CHI-1", "DAL-2", "ATL-3", "SEA-4"][i % 4]}`,
      hazmat: i % 10 === 0,
    });
  }
  return JSON.stringify(arr, null, 2);
}

function csvFixture(n = 40) {
  const rows = ["sku,name,category,warehouse,currency"];
  const cats = ["packaging", "equipment", "labels"];
  for (let i = 0; i < n; i++) {
    rows.push(`NW-${1000 + i},Item ${i},${cats[i % 3]},WH-${["CHI-1", "DAL-2"][i % 2]},USD`);
  }
  return rows.join("\n") + "\n";
}

function logsFixture(n = 40) {
  const lines = [];
  for (let i = 0; i < n; i++) {
    lines.push(
      `2026-07-${String(10 + (i % 5)).padStart(2, "0")}T12:00:${String(i % 60).padStart(2, "0")}Z ` +
      `INFO [api] reqId=${7000 + i} handled request path=/v1/items status=200 ms=${10 + (i % 90)}`
    );
  }
  return lines.join("\n") + "\n";
}

const PROSE = "This is ordinary prose.\nIt has no repeated record structure at all.\n" +
  "ctxfold should decline it.\nNothing here is tabular.\nJust sentences.\n";

function quotedCsvFixture() {
  const rows = ['sku,name,notes'];
  for (let i = 0; i < 10; i++) rows.push(`NW-${i},"Item ${i}, deluxe",plain`);
  return rows.join("\n") + "\n";
}

// ---------------------------------------------------------------------------
// Invariant 1: composition is exact — categories sum to the profiled size.

function compositionSum(p) {
  return p.composition.reduce((s, c) => s + c.chars, 0);
}

test("json composition sums exactly to input size", () => {
  const text = jsonFixture();
  const p = profile(text);
  assert.strictEqual(p.format, "json");
  assert.ok(!p.compositionNormalized, "pretty-printed fixture should not need normalization");
  assert.strictEqual(compositionSum(p), text.length);
});

test("csv composition sums exactly to input size", () => {
  const text = csvFixture();
  const p = profile(text);
  assert.strictEqual(p.format, "csv");
  assert.strictEqual(compositionSum(p), text.length);
});

test("logs composition sums exactly to input size", () => {
  const text = logsFixture();
  const p = profile(text);
  assert.strictEqual(p.format, "logs");
  assert.strictEqual(compositionSum(p), text.length);
});

test("composition percentages sum to ~100%", () => {
  for (const text of [jsonFixture(), csvFixture(), logsFixture()]) {
    const p = profile(text);
    const pct = p.composition.reduce((s, c) => s + c.pct, 0);
    assert.ok(Math.abs(pct - 1) < 1e-9, `pct sum ${pct} for ${p.format}`);
  }
});

// ---------------------------------------------------------------------------
// Invariant 2: profile never promises more than compress() delivers — the
// reported ratios ARE compress()'s ratios on the same input.

test("profile foldable ratio equals compress() ratio", () => {
  for (const text of [jsonFixture(), csvFixture(), logsFixture()]) {
    const p = profile(text);
    const { stats } = compress(text);
    assert.strictEqual(p.foldable[0].tokenRatio, stats.tokenRatio, p.format);
    assert.strictEqual(p.foldable[0].tokensAfter, stats.tokensAfter, p.format);
  }
});

test("dictionary line only appears when it actually improves on plain fold", () => {
  const p = profile(jsonFixture());
  const dictLine = p.foldable.find((f) => f.label === "+ --dictionary");
  if (dictLine) {
    const plain = compress(jsonFixture());
    const dict = compress(jsonFixture(), { dictionary: true });
    assert.ok(dict.stats.tokensAfter < plain.stats.tokensAfter);
    assert.match(dictLine.note, /readability tradeoff/);
  }
});

// ---------------------------------------------------------------------------
// Invariant 3: honesty flags per format.

test("csv verdict always carries the pipeline-only warning", () => {
  const p = profile(csvFixture());
  assert.match(p.foldable[0].note, /pipeline-only/i);
  assert.match(p.verdict, /decompress\(\)/);
});

test("json and logs claims cite validated direct readability", () => {
  assert.match(profile(jsonFixture()).foldable[0].note, /direct-readable/);
  assert.match(profile(logsFixture()).foldable[0].note, /direct-readable/);
});

test("token totals are marked estimated unless a tokenizer is passed", () => {
  assert.strictEqual(profile(jsonFixture()).tokensExact, false);
  const p = profile(jsonFixture(), { countTokens: (s) => s.split(/\s+/).length });
  assert.strictEqual(p.tokensExact, true);
});

// ---------------------------------------------------------------------------
// Invariant 4: unrecognized input produces the no-op verdict plus reasons.

test("prose declines with an explanation", () => {
  const p = profile(PROSE);
  assert.strictEqual(p.format, "none");
  assert.match(p.verdict, /nothing to fold/);
  assert.ok(Array.isArray(p.reasons) && p.reasons.length > 0);
});

test("quoted csv declines and the reason says why", () => {
  const p = profile(quotedCsvFixture());
  assert.strictEqual(p.format, "none");
  assert.ok(p.reasons.some((r) => /quote/i.test(r)), p.reasons.join("; "));
});

// ---------------------------------------------------------------------------
// Renderer sanity: report contains the essentials and renders for all shapes.

test("rendered report contains format, composition, and verdict", () => {
  for (const text of [jsonFixture(), csvFixture(), logsFixture(), PROSE]) {
    const out = renderProfile(profile(text));
    assert.match(out, /\[ctxfold profile\]/);
    assert.match(out, /verdict:/);
  }
});
