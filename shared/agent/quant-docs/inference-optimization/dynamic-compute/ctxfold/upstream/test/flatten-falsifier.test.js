"use strict";

// PRE-REGISTERED FALSIFIER — committed before any flattening model code.
//
// Contract under test: one level of JSON nesting folds into `user.name`-style
// columns without making the encoding ambiguous.
//
// Two things every case asserts, and it must be BOTH:
//   (1) DISPOSITION — did the record fold, or was it refused to verbatim?
//   (2) FIDELITY    — does the payload round-trip?
//
// (2) alone is vacuous. An implementation that refuses to flatten anything
// passes a pure losslessness assertion trivially, because compress() falls back
// to returning the input untouched. The disposition assertions are what make
// this a falsifier rather than a tautology.
//
// FIDELITY IS CANONICAL-EXACT, NOT BYTE-EXACT. json.verify() compares
// JSON.stringify(JSON.parse(original)) === restored, and the README documents
// JSON losslessness as value-identical. A byte-exact assertion fails on the
// fixture's own indentation before flattening is reached, so it would falsify
// nothing about this increment.

const test = require("node:test");
const assert = require("node:assert");
const { compress, decompress } = require("../src/index");

const canon = (s) => JSON.stringify(JSON.parse(s));
const N = 24; // enough conforming rows to clear the modal-signature floor and pay for the legend

function specOf(out) {
  return JSON.parse(out.split("\n")[1].slice("spec ".length));
}
// Column paths as the decoder sees them: explicit `p` when present, else [k].
function pathsOf(out) {
  return specOf(out).cols.map((c) => (c.p ? c.p : [c.k]));
}
function displayNamesOf(out) {
  return specOf(out).cols.map((c) => c.k);
}
function verbatimIdx(out) {
  return Object.keys(specOf(out).verbatim).map(Number).sort((a, b) => a - b);
}
function folds(text) {
  const { text: out, stats } = compress(text);
  return { out, stats, folded: stats.encoder === "json" };
}
function assertLossless(text, out) {
  assert.strictEqual(decompress(out), canon(text), "round-trip must be canonical-exact");
}

// --- FX1  literal dotted top-level key, no nesting present -------------------
// `user.name` here is ONE key with a dot in it. Must stay a length-1 path.
test("FX1: literal dotted top-level key folds as a length-1 path", () => {
  const arr = [];
  for (let i = 0; i < N; i++) arr.push({ id: i, "user.name": `nm-${i}`, region: `r${i % 3}` });
  const text = JSON.stringify(arr);
  const { out, folded } = folds(text);
  assert.ok(folded, "expected the json encoder to fire");
  assert.deepStrictEqual(pathsOf(out), [["id"], ["user.name"], ["region"]]);
  assertLossless(text, out);
});

// --- FX2  one level of nesting — the feature itself --------------------------
test("FX2: one level of nesting folds to a length-2 path shown as user.name", () => {
  const arr = [];
  for (let i = 0; i < N; i++) arr.push({ id: i, user: { name: `nm-${i}`, role: `r${i % 3}` } });
  const text = JSON.stringify(arr);
  const { out, folded } = folds(text);
  assert.ok(folded, "expected the json encoder to fire");
  assert.deepStrictEqual(pathsOf(out), [["id"], ["user", "name"], ["user", "role"]]);
  assert.deepStrictEqual(displayNamesOf(out), ["id", "user.name", "user.role"]);
  assert.deepStrictEqual(verbatimIdx(out), [], "no record should be refused here");
  assertLossless(text, out);
});

// --- FX3  the collision: literal `user.name` AND nested user.name, same record
// Decode could disambiguate via paths, but the MODEL reading the folded text
// would see two identical column headers. Readability is the product, so refuse.
test("FX3: literal and nested paths that would render the same header refuse to flatten", () => {
  const arr = [];
  for (let i = 0; i < N; i++) arr.push({ id: i, "user.name": `lit-${i}`, user: { name: `nest-${i}` } });
  const text = JSON.stringify(arr);
  const { out, stats } = folds(text);
  if (stats.encoder === "json") {
    const names = displayNamesOf(out);
    assert.strictEqual(new Set(names).size, names.length, "no two columns may share a display name");
    assert.ok(!names.includes("user.name") || pathsOf(out).every((p) => p.length === 1),
      "must not emit a flattened user.name alongside the literal one");
  }
  assertLossless(text, out);
});

// --- FX4  empty nested object ------------------------------------------------
// `{}` yields zero columns, so no column layout can distinguish it from an
// absent key. Refuse. This is also the status quo in 0.2.2 (empty object is
// "complex"), so refusing is not a regression.
test("FX4: empty nested object is refused, not flattened away", () => {
  const arr = [];
  for (let i = 0; i < N; i++) arr.push({ id: i, user: {}, region: `r${i % 3}` });
  const text = JSON.stringify(arr);
  const { out, stats } = folds(text);
  if (stats.encoder === "json") {
    assert.ok(pathsOf(out).every((p) => p[0] !== "user" || p.length === 1),
      "an empty object must not contribute flattened columns");
  }
  assertLossless(text, out);
});

// --- FX5  explicit null vs absent key ---------------------------------------
// null is a value and folds; an absent key changes the signature and goes
// verbatim. The two must not converge on the same encoding.
test("FX5: explicit null folds, absent key goes verbatim", () => {
  const arr = [];
  for (let i = 0; i < N; i++) {
    if (i === 9) arr.push({ id: i, region: `r${i % 3}` });          // absent
    else arr.push({ id: i, user: i === 4 ? null : { name: `nm-${i}` }, region: `r${i % 3}` });
  }
  const text = JSON.stringify(arr);
  const { out, folded } = folds(text);
  assert.ok(folded, "expected the json encoder to fire");
  assert.ok(verbatimIdx(out).includes(9), "the absent-key record must be refused");
  assert.ok(!verbatimIdx(out).includes(0), "ordinary records must still fold");
  assertLossless(text, out);
});

// --- FX6  same key holding an object in one record and a scalar in another ---
test("FX6: mixed-shape key splits — modal shape folds, the other goes verbatim", () => {
  const arr = [];
  for (let i = 0; i < N; i++) arr.push({ id: i, user: { name: `nm-${i}` } });
  arr.splice(7, 0, { id: 999, user: "bob" });
  const text = JSON.stringify(arr);
  const { out, folded } = folds(text);
  assert.ok(folded, "expected the json encoder to fire");
  assert.ok(verbatimIdx(out).includes(7), "the scalar-valued record must be refused");
  assertLossless(text, out);
});

// --- FX7  dotted INNER key vs genuine two-level nesting ----------------------
// The case that decides whether the next increment needs a format break.
// {user:{"first.name":v}} and {user:{first:{name:v}}} both render as
// user.first.name. They must differ in the spec.
test("FX7: dotted inner key is a length-2 path, distinct from two-level nesting", () => {
  const inner = [];
  for (let i = 0; i < N; i++) inner.push({ id: i, user: { "first.name": `nm-${i}` } });
  const innerText = JSON.stringify(inner);
  const a = folds(innerText);
  assert.ok(a.folded, "expected the json encoder to fire on the dotted inner key");
  assert.deepStrictEqual(pathsOf(a.out), [["id"], ["user", "first.name"]]);
  assert.deepStrictEqual(displayNamesOf(a.out), ["id", "user.first.name"]);
  assertLossless(innerText, a.out);

  // Two-level is out of scope for v0.3.0: it must be refused, never flattened
  // into a path that collides with the length-2 encoding above.
  const two = [];
  for (let i = 0; i < N; i++) two.push({ id: i, user: { first: { name: `nm-${i}` } } });
  const twoText = JSON.stringify(two);
  const b = folds(twoText);
  if (b.stats.encoder === "json") {
    assert.ok(pathsOf(b.out).every((p) => p.length <= 1),
      "two-level nesting must not be flattened by the one-level increment");
  }
  assertLossless(twoText, b.out);
});

// --- FX8  no behaviour change on inputs with no nesting ----------------------
// 0.3.0 must be byte-identical to 0.2.2 whenever nothing nests, so the feature
// cannot silently reshape existing payloads.
test("FX8: inputs with no nesting emit no path field and stay at spec v1", () => {
  const arr = [];
  for (let i = 0; i < N; i++) arr.push({ id: i, name: `nm-${i}`, region: `r${i % 3}` });
  const text = JSON.stringify(arr);
  const { out, folded } = folds(text);
  assert.ok(folded, "expected the json encoder to fire");
  assert.ok(specOf(out).cols.every((c) => c.p === undefined), "no `p` on unnested columns");
  assert.strictEqual(specOf(out).v, 1, "unnested payloads stay readable by older decoders");
  assertLossless(text, out);
});

// --- FX9  version gate -------------------------------------------------------
// decode() in 0.2.2 destructures {n, cols, verbatim, wrap} and never reads
// spec.v, so it silently mis-decodes any future format. Flattening is the first
// increment that makes that reachable, so the gate ships with it.
test("FX9: a payload declaring a newer spec version is rejected, not mis-decoded", () => {
  const arr = [];
  for (let i = 0; i < N; i++) arr.push({ id: i, name: `nm-${i}`, region: `r${i % 3}` });
  const { text: out } = compress(JSON.stringify(arr));
  const lines = out.split("\n");
  const sp = JSON.parse(lines[1].slice("spec ".length));
  sp.v = 99;
  lines[1] = "spec " + JSON.stringify(sp);
  assert.throws(() => decompress(lines.join("\n")), /version/i,
    "an unknown spec version must throw, not decode on a guess");
});
