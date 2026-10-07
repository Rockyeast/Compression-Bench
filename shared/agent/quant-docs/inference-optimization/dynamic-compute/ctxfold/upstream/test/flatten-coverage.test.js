"use strict";

// Coverage added after the v0.3.0 mutation pass. SEPARATE FILE ON PURPOSE:
// test/flatten-falsifier.test.js was pre-registered and committed before any
// model code, and editing it after seeing the implementation's results would
// destroy the thing pre-registration buys. These are additional cases, not
// revisions.
//
// Each case kills a mutant that survived the full suite. The mutant is named.
// Every one of the four needs a record MIXED IN with foldable records — the
// pure fixtures in the falsifier are all rescued by compress()'s verify()
// safety net, which declines the encoder wholesale and leaves round-trip
// assertions trivially true. That is the same vacuity trap the falsifier's own
// header warns about, one level down.

const test = require("node:test");
const assert = require("node:assert");
const { compress, decompress } = require("../src/index");

const canon = (s) => JSON.stringify(JSON.parse(s));
const specOf = (out) => JSON.parse(out.split("\n")[1].slice("spec ".length));
const verbatimIdx = (out) => Object.keys(specOf(out).verbatim).map(Number).sort((a, b) => a - b);

// Kills M1: `if (inner.length === 0) return null`.
// Without it an empty nested object contributes zero columns, so {user:{}} and
// a record with no `user` at all share a signature and the empty object is lost.
// verify() catches the loss and declines the whole input — so the visible
// damage is a foldable input silently degrading to a no-op.
test("MX1: an empty nested object is refused per-record, not by declining the input", () => {
  const arr = [];
  for (let i = 0; i < 24; i++) {
    arr.push(i % 6 === 0 ? { id: i, user: {}, region: "r" } : { id: i, region: `r${i % 3}` });
  }
  const text = JSON.stringify(arr);
  const { text: out, stats } = compress(text);
  assert.strictEqual(stats.encoder, "json", "the foldable majority must still fold");
  assert.deepStrictEqual(verbatimIdx(out), [0, 6, 12, 18], "only the empty-object records are refused");
  assert.strictEqual(decompress(out), canon(text));
});

// Kills M6: `v: nested ? 2 : 1` pinned to 1.
// FX8 fixes the unnested half of this; nothing pinned the nested half. A nested
// payload mislabelled v1 is exactly what a 0.2.2 decoder will read without
// looking for `p`, rebuilding every nested column as a literal dotted key.
test("MX2: a payload containing flattened columns declares spec v2", () => {
  const arr = [];
  for (let i = 0; i < 24; i++) arr.push({ id: i, user: { name: `nm-${i}`, role: `r${i % 3}` } });
  const text = JSON.stringify(arr);
  const { text: out, stats } = compress(text);
  assert.strictEqual(stats.encoder, "json");
  assert.strictEqual(specOf(out).v, 2, "nested payloads must not claim the older version");
  assert.ok(specOf(out).cols.some((c) => c.p), "sanity: this fixture really is nested");
  assert.strictEqual(decompress(out), canon(text));
});

// Kills M7: the two-level refusal in recordPaths.
// The refusal only bites when the deep record shares the modal shape's INNER
// key name — otherwise its path list differs and the modal-signature partition
// sends it verbatim for an unrelated reason. FX7 uses a whole-array fixture and
// so never reaches this.
test("MX3: two levels under the modal inner key name is refused, not encoded as a cell", () => {
  const arr = [];
  for (let i = 0; i < 24; i++) {
    arr.push(i === 5 ? { id: i, user: { name: { deep: 1 } } } : { id: i, user: { name: `nm-${i}` } });
  }
  const text = JSON.stringify(arr);
  const { text: out, stats } = compress(text);
  assert.strictEqual(stats.encoder, "json", "one deep record must not poison the whole input");
  assert.deepStrictEqual(verbatimIdx(out), [5]);
  assert.strictEqual(decompress(out), canon(text));
});

// Kills M8: the `|| layout(arr, false)` fallback.
// Reachable only when the COLLIDING shape is the modal one — the collision
// check reads the modal signature, so a colliding minority is already sent
// verbatim by the partition and never triggers it.
test("MX4: a colliding modal shape falls back to the flat-only layout", () => {
  const arr = [];
  for (let i = 0; i < 30; i++) arr.push({ i, "a.b": 1, a: { b: 2 } }); // modal, and collides
  for (let i = 0; i < 26; i++) {
    arr.push({
      sku: `NORTHWIND-CATALOG-ITEM-${1000 + i}`,
      description: `A long and highly repetitive description field number ${i}`,
      warehouse: "CHICAGO-DISTRIBUTION-CENTER-1",
      category: "industrial-packaging-supplies",
    });
  }
  const text = JSON.stringify(arr);
  const { text: out, stats } = compress(text);
  assert.strictEqual(stats.encoder, "json", "refusing to flatten must not mean refusing the input");
  assert.ok(specOf(out).cols.every((c) => c.p === undefined), "the fallback layout flattens nothing");
  assert.strictEqual(specOf(out).v, 1, "and therefore stays at spec v1");
  assert.deepStrictEqual(verbatimIdx(out), Array.from({ length: 30 }, (_, i) => i));
  assert.strictEqual(decompress(out), canon(text));
});
