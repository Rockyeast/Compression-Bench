"use strict";

// JSON-array encoder. Detects a top-level JSON array of objects, and re-encodes
// as a clean column table: column paths + per-column type declared ONCE in the
// header, then one tab-separated row of bare values per object. Objects that
// don't fit the modal schema are kept verbatim. Reversible; losslessness is
// value-level (see verify()).
//
// v0.3.0 — ONE LEVEL OF NESTING. {user:{name:"a"}} folds into a column shown as
// `user.name`. The dotted string is a DISPLAY affordance for the model reading
// the folded text; the authoritative form is the path ARRAY carried in the spec
// as `p`. An array is closed under nesting depth, so ["user","first.name"] (a
// dotted inner key) and ["user","first","name"] (genuine two-level nesting) are
// distinct values and can never collide. Nothing about a deeper increment
// requires a format break here.
//
// `p` is omitted whenever it would equal [k], so any input with no nesting
// produces byte-identical output to v0.2.2 and stays at spec v1.

function scalarKind(v) {
  if (v === null) return "null";
  const t = typeof v;
  if (t === "string" || t === "number" || t === "boolean") return t;
  return "complex"; // object/array — not a leaf value
}

function isPlainObject(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function detect(text) {
  const t = text.trim();
  if (t.length <= 200 || !/\[\s*\{/.test(t)) return false;
  return t[0] === "[" || t[0] === "{"; // bare array OR object wrapping an array
}

// From a parsed top-level value, find the records array to encode.
// Returns { arr, wrap } where wrap is null for a bare array, or
// { order, arrayKey, rest } for an object wrapping exactly one records array.
function extractArray(parsed) {
  if (Array.isArray(parsed)) return { arr: parsed, wrap: null };
  if (!parsed || typeof parsed !== "object") return null;

  const order = Object.keys(parsed);
  const arrayKeys = order.filter((k) => {
    const v = parsed[k];
    if (!Array.isArray(v) || v.length < 4) return false;
    const objs = v.filter((e) => e && typeof e === "object" && !Array.isArray(e)).length;
    return objs >= v.length * 0.6;
  });
  if (arrayKeys.length !== 1) return null; // 0 or ambiguous -> decline

  const arrayKey = arrayKeys[0];
  const rest = {};
  for (const k of order) if (k !== arrayKey) rest[k] = parsed[k];
  return { arr: parsed[arrayKey], wrap: { order, arrayKey, rest } };
}

// The column layout a single record wants, as a list of paths in source order.
// Returns null when the record cannot be represented — the caller keeps it
// verbatim, which is always available and always lossless.
//
// Refusals, and why each one is a refusal rather than an encoding:
//   - an empty nested object yields ZERO columns, so no column layout can
//     distinguish {user:{}} from an absent `user`. (Already the v0.2.2
//     behaviour: an empty object is "complex".)
//   - a nested ARRAY has no column form at all.
//   - a value nested two levels down is out of scope for this increment. It
//     must not be flattened into a path that would collide with the one-level
//     encoding of a dotted inner key.
function recordPaths(obj, flatten) {
  if (!isPlainObject(obj)) return null;
  const paths = [];
  for (const k of Object.keys(obj)) {
    const v = obj[k];
    if (scalarKind(v) !== "complex") { paths.push([k]); continue; }
    if (!flatten) return null;
    if (!isPlainObject(v)) return null;          // array
    const inner = Object.keys(v);
    if (inner.length === 0) return null;         // empty nested object
    for (const ik of inner) {
      if (scalarKind(v[ik]) === "complex") return null; // two-level, or array
      paths.push([k, ik]);
    }
  }
  return paths;
}

function valueAt(obj, path) {
  return path.length === 1 ? obj[path[0]] : obj[path[0]][path[1]];
}

// Per-cell tagged encoding, used only for "mixed"-type columns.
function encodeCell(v) {
  switch (scalarKind(v)) {
    case "string": return "s" + JSON.stringify(v);
    case "number": return "n" + String(v);
    case "boolean": return v ? "b1" : "b0";
    case "null": return "z";
    default: return null;
  }
}
function decodeCell(cell) {
  const tag = cell[0], body = cell.slice(1);
  if (tag === "s") return JSON.parse(body);
  if (tag === "n") return Number(body);
  if (tag === "b") return body === "1";
  if (tag === "z") return null;
  throw new Error("bad cell tag");
}

// Decide a column's encoding type from its values.
function columnType(values) {
  const kinds = new Set(values.map(scalarKind));
  if (kinds.size === 1) {
    const only = kinds.values().next().value;
    if (only === "string") {
      // Bare strings are only safe if they contain no tab/newline (our delims).
      if (values.every((v) => v.indexOf("\t") === -1 && v.indexOf("\n") === -1)) return "s";
      return "mixed";
    }
    if (only === "number") return "n";
    if (only === "boolean") return "b";
    if (only === "null") return "z";
  }
  return "mixed";
}

function encodeByType(v, t) {
  switch (t) {
    case "s": return v;
    case "n": return String(v);
    case "b": return v ? "1" : "0";
    case "z": return "";
    default: return encodeCell(v);
  }
}
function decodeByType(cell, t) {
  switch (t) {
    case "s": return cell;
    case "n": return Number(cell);
    case "b": return cell === "1";
    case "z": return null;
    default: return decodeCell(cell);
  }
}

// Build a dictionary for a low-cardinality column: distinct values become small
// integer codes. Returns { dict, index } or null. Only fires when coding the
// column actually saves bytes (so enabling dictionaries can never do worse than
// bare). Lossless: each cell stores the code, the dict maps code -> value.
function buildDictionary(values, { maxDistinct = 256 } = {}) {
  const index = new Map();
  const dict = [];
  for (const v of values) {
    if (!index.has(v)) { index.set(v, dict.length); dict.push(v); }
  }
  const d = dict.length;
  if (d < 2 || d > maxDistinct || d >= values.length) return null;

  // Estimate net savings: rows pay (value length -> code length); the dictionary
  // header costs ~"idx=value, " per entry once.
  const avgVal = dict.reduce((s, v) => s + v.length, 0) / d;
  const avgCode = values.reduce((s, v) => s + String(index.get(v)).length, 0) / values.length;
  const rowSavings = values.length * (avgVal - avgCode);
  const headerCost = dict.reduce((s, v, i) => s + String(i).length + 1 + v.length + 2, 0) + 16;
  if (rowSavings - headerCost <= 0) return null;

  return { dict, index };
}

// Choose the modal column layout and partition the array against it.
// Returns { paths, conforming, verbatim } or null.
function layout(arr, flatten) {
  const sigCount = new Map(), sigPaths = new Map(), perRecord = new Array(arr.length);

  for (let i = 0; i < arr.length; i++) {
    const paths = recordPaths(arr[i], flatten);
    perRecord[i] = paths;
    if (!paths) continue;
    const sig = JSON.stringify(paths);
    sigCount.set(sig, (sigCount.get(sig) || 0) + 1);
    if (!sigPaths.has(sig)) sigPaths.set(sig, paths);
  }
  if (sigCount.size === 0) return null;

  let bestSig = null, bestN = -1;
  for (const [sig, n] of sigCount) if (n > bestN) { bestSig = sig; bestN = n; }
  if (bestN < 4) return null;
  const paths = sigPaths.get(bestSig);

  // THE COLLISION REFUSAL. decode() reads `p`, so it could tell a literal
  // dotted key from a flattened path — but the model reading the folded text
  // sees only the rendered header, and two identical headers are ambiguous to
  // it. Model-readability is the product, so refuse and let the caller retry
  // without flattening.
  const names = paths.map((p) => p.join("."));
  if (new Set(names).size !== names.length) return null;

  const conforming = [], verbatim = {};
  for (let i = 0; i < arr.length; i++) {
    if (perRecord[i] && JSON.stringify(perRecord[i]) === bestSig) conforming.push(i);
    else verbatim[i] = JSON.stringify(arr[i]);
  }
  if (conforming.length < 4) return null;

  return { paths, conforming, verbatim };
}

function encode(text, opts = {}) {
  let parsed;
  try { parsed = JSON.parse(text); } catch { return { ok: false }; }
  const extracted = extractArray(parsed);
  if (!extracted) return { ok: false };
  const { arr, wrap } = extracted;
  if (!Array.isArray(arr) || arr.length < 4) return { ok: false };

  // Flatten if we can; fall back to the flat-only layout if flattening would
  // produce colliding headers.
  const plan = layout(arr, true) || layout(arr, false);
  if (!plan) return { ok: false };
  const { paths, conforming, verbatim } = plan;
  const names = paths.map((p) => p.join("."));

  // Per-column encoding plan. Default: type-based bare cells. With opts.dictionary,
  // low-cardinality string columns become dictionary-coded (values -> small codes).
  const plans = paths.map((p) => {
    const vals = conforming.map((i) => valueAt(arr[i], p));
    const t = columnType(vals);
    if (opts.dictionary && t === "s") {
      const dic = buildDictionary(vals);
      if (dic) return { t: "d", dict: dic.dict, index: dic.index };
    }
    return { t };
  });

  const rows = conforming.map((i) =>
    paths.map((p, c) => {
      const pl = plans[c];
      return pl.t === "d" ? String(pl.index.get(valueAt(arr[i], p))) : encodeByType(valueAt(arr[i], p), pl.t);
    }).join("\t")
  );

  const nested = paths.some((p) => p.length > 1);
  const cols = paths.map((p, c) => {
    const col = { k: names[c] };
    if (p.length > 1) col.p = p;
    if (plans[c].t === "d") { col.t = "d"; col.dict = plans[c].dict; }
    else col.t = plans[c].t;
    return col;
  });

  // Emit the LOWEST spec version that expresses this payload, so unnested
  // output stays readable by decoders that predate flattening.
  const spec = { v: nested ? 2 : 1, n: arr.length, cols, verbatim, wrap };

  const dictNote = plans.some((pl) => pl.t === "d")
    ? " Dictionary columns (code=value): " +
      names.map((k, c) => plans[c].t === "d"
        ? `${k} {${plans[c].dict.map((v, idx) => `${idx}=${v}`).join(", ")}}`
        : null).filter(Boolean).join("; ") + "."
    : "";

  const nestNote = nested
    ? " Nested fields, written parent.child: " +
      names.filter((k, c) => paths[c].length > 1).join(", ") + "."
    : "";

  const wrapNote = wrap
    ? `a JSON object with keys [${wrap.order.join(", ")}], where "${wrap.arrayKey}" is the array below`
    : `a JSON array`;
  const legend =
    `legend: ${wrapNote} of ${arr.length} objects as a table. Columns: ` +
    `${names.join(", ")}. Each row is tab-separated values in that order, in array order. ` +
    `Values are bare unless noted.${nestNote}${dictNote}`;

  const out =
    "\u27e6cf/json v1\u27e7\n" +
    "spec " + JSON.stringify(spec) + "\n" +
    legend + "\n" +
    "cols: " + names.join("\t") + "\n" +
    "rows:\n" +
    rows.join("\n");

  return { ok: true, encoded: out };
}

const MAX_SPEC_VERSION = 2;

function decode(encoded) {
  const nl = encoded.indexOf("\n");
  if (encoded.slice(0, nl) !== "\u27e6cf/json v1\u27e7") throw new Error("bad magic");
  let rest = encoded.slice(nl + 1);

  const specEnd = rest.indexOf("\n");
  const spec = JSON.parse(rest.slice("spec ".length, specEnd));
  rest = rest.slice(specEnd + 1);

  // Version gate. Without it a payload from a later format decodes on a guess:
  // a `p` field this decoder didn't know about would be dropped and every
  // nested column silently rebuilt as a literal dotted key.
  if (!Number.isInteger(spec.v) || spec.v < 1 || spec.v > MAX_SPEC_VERSION) {
    throw new Error(
      "ctxfold/json: unsupported spec version " + spec.v +
      " (this build reads up to " + MAX_SPEC_VERSION + ")"
    );
  }

  const marker = "\nrows:\n";
  const rowsBlock = rest.slice(rest.indexOf(marker) + marker.length);
  const rowLines = rowsBlock.length ? rowsBlock.split("\n") : [];

  const { n, cols, verbatim, wrap } = spec;
  const out = new Array(n);
  let rp = 0;
  for (let i = 0; i < n; i++) {
    if (Object.prototype.hasOwnProperty.call(verbatim, i)) {
      out[i] = JSON.parse(verbatim[i]);
      continue;
    }
    const line = rowLines[rp++];
    if (line === undefined) throw new Error("ctxfold/json: missing row (rows fewer than schema declares)");
    const cells = line.split("\t");
    if (cells.length !== cols.length) throw new Error("ctxfold/json: row has " + cells.length + " cells, schema declares " + cols.length + " columns");
    const obj = {};
    for (let c = 0; c < cols.length; c++) {
      const col = cols[c];
      let value;
      if (col.t === "d") {
        const code = Number(cells[c]);
        if (!Number.isInteger(code) || code < 0 || code >= col.dict.length) {
          throw new Error("ctxfold/json: dictionary code '" + cells[c] + "' out of range for column '" + col.k + "'");
        }
        value = col.dict[code];
      } else {
        value = decodeByType(cells[c], col.t);
      }
      // `p` absent means the column IS a top-level key, dots and all.
      const path = col.p || [col.k];
      if (path.length === 1) {
        obj[path[0]] = value;
      } else {
        // Columns sharing a parent are contiguous by construction — they were
        // emitted from one source object in its own key order — so creating the
        // parent on first sight reproduces the original key order.
        if (!Object.prototype.hasOwnProperty.call(obj, path[0])) obj[path[0]] = {};
        obj[path[0]][path[1]] = value;
      }
    }
    out[i] = obj;
  }

  if (wrap) {
    const result = {};
    for (const k of wrap.order) result[k] = k === wrap.arrayKey ? out : wrap.rest[k];
    return JSON.stringify(result);
  }
  return JSON.stringify(out);
}

// Value-level losslessness: decode reproduces the same DATA when parsed.
function verify(original, restored) {
  try { return JSON.stringify(JSON.parse(original)) === restored; }
  catch { return false; }
}

module.exports = { name: "json", detect, encode, decode, verify };
