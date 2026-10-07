"use strict";

const { factorColumn } = require("../affix");

// Recognize the atoms that make a line "templated".
const RE_TS = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?$/;
const LEVELS = new Set(["TRACE", "DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"]);
const RE_KV = /^([A-Za-z_][\w.\-]*)=/;
const RE_BRACKET = /^\[.*\]$/;
const RE_NUM = /^-?\d+(?:\.\d+)?$/;

function fieldType(f) {
  if (RE_TS.test(f)) return "ts";
  if (LEVELS.has(f)) return "lvl";
  if (RE_BRACKET.test(f)) return "br";
  if (RE_KV.test(f)) return "kv";
  if (RE_NUM.test(f)) return "num";
  return "w";
}

// How many leading fields are "structured" (ts / level / [..] / key=value).
function leadLen(fields) {
  let i = 0;
  while (i < fields.length) {
    const t = fieldType(fields[i]);
    if (t === "ts" || t === "lvl" || t === "br" || t === "kv") i++;
    else break;
  }
  return i;
}

// How many trailing fields are key=value or numeric.
function trailLen(fields, stopBefore) {
  let i = 0;
  while (fields.length - 1 - i >= stopBefore) {
    const t = fieldType(fields[fields.length - 1 - i]);
    if (t === "kv" || t === "num") i++;
    else break;
  }
  return i;
}

function detect(text) {
  const lines = text.split("\n").filter((l) => l.length > 0);
  if (lines.length < 8) return false;
  let structured = 0;
  for (const l of lines) {
    const f = l.split(" ");
    if (leadLen(f) >= 2) structured++;
  }
  return structured / lines.length >= 0.6;
}

function modal(pairs) {
  const counts = new Map();
  for (const p of pairs) counts.set(p, (counts.get(p) || 0) + 1);
  let best = null, bestN = -1;
  for (const [p, n] of counts) if (n > bestN) { best = p; bestN = n; }
  return best;
}

function encode(text) {
  const eol = text.endsWith("\n");
  const raw = text.split("\n");
  // If eol, the final split element is "" — drop it from the logical line list.
  const lines = eol ? raw.slice(0, -1) : raw;
  const n = lines.length;

  // Decide template shape (L lead cols, T trail cols) from the modal signature.
  const sigs = [];
  for (const l of lines) {
    const f = l.split(" ");
    const L = leadLen(f);
    const T = trailLen(f, L);
    sigs.push(`${L},${T}`);
  }
  const [L, T] = modal(sigs).split(",").map(Number);
  if (L + T === 0) return { ok: false };

  // Partition lines into conforming rows vs verbatim.
  const conformingIdx = [];
  const verbatim = {};
  const leadCols = Array.from({ length: L }, () => []);
  const trailCols = Array.from({ length: T }, () => []);
  const middles = [];

  for (let i = 0; i < n; i++) {
    const f = lines[i].split(" ");
    if (sigs[i] === `${L},${T}` && f.length >= L + T) {
      conformingIdx.push(i);
      for (let c = 0; c < L; c++) leadCols[c].push(f[c]);
      for (let c = 0; c < T; c++) trailCols[c].push(f[f.length - T + c]);
      middles.push(f.slice(L, f.length - T).join(" "));
    } else {
      verbatim[i] = lines[i];
    }
  }
  if (conformingIdx.length < 8) return { ok: false };

  const leadF = leadCols.map(factorColumn);
  const trailF = trailCols.map(factorColumn);

  // Derive a readable name per column from its data (key= keys, time, level...).
  const used = new Map();
  const nameOf = (values, idx) => {
    const v = values[0] || "";
    const t = fieldType(v);
    let base;
    if (t === "kv") { const m = v.match(RE_KV); base = m ? m[1] : "kv"; }
    else if (t === "ts") base = "time";
    else if (t === "lvl") base = "level";
    else if (t === "br") base = "scope";
    else if (t === "num") base = "num";
    else base = "field";
    const seen = used.get(base) || 0;
    used.set(base, seen + 1);
    return seen === 0 ? base : `${base}${idx}`;
  };
  const leadNames = leadCols.map((c, i) => nameOf(c, i));
  const trailNames = trailCols.map((c, i) => nameOf(c, L + i));
  const colNames = [...leadNames, ...trailNames, "message"];

  const spec = {
    v: 1, n, eol, L, T,
    lead: leadF.map((c) => ({ pre: c.prefix, suf: c.suffix })),
    trail: trailF.map((c) => ({ pre: c.prefix, suf: c.suffix })),
    verbatim,
  };

  // Build row lines: lead middles, trail middles, then the free-text middle.
  const rows = [];
  for (let r = 0; r < conformingIdx.length; r++) {
    const cells = [];
    for (let c = 0; c < L; c++) cells.push(leadF[c].middles[r]);
    for (let c = 0; c < T; c++) cells.push(trailF[c].middles[r]);
    const mid = middles[r];
    rows.push(mid.length ? cells.join(" ") + " " + mid : cells.join(" "));
  }

  const legend =
    `legend: each row is space-separated as: ${colNames.slice(0, -1).join(" ")} ` +
    `then the message (everything after the first ${L + T} values).`;

  const out =
    "\u27e6cf/logs v1\u27e7\n" +
    "spec " + JSON.stringify(spec) + "\n" +
    legend + "\n" +
    "cols: " + colNames.join(" ") + "\n" +
    "rows:\n" +
    rows.join("\n");

  return { ok: true, encoded: out };
}

function decode(encoded) {
  const nl = encoded.indexOf("\n");
  const magic = encoded.slice(0, nl);
  if (magic !== "\u27e6cf/logs v1\u27e7") throw new Error("bad magic");
  let rest = encoded.slice(nl + 1);

  const specLineEnd = rest.indexOf("\n");
  const specLine = rest.slice(0, specLineEnd);
  const spec = JSON.parse(specLine.slice("spec ".length));
  rest = rest.slice(specLineEnd + 1);

  const marker = "\nrows:\n";
  const mi = rest.indexOf(marker);
  const rowsBlock = rest.slice(mi + marker.length);
  const rows = rowsBlock.length ? rowsBlock.split("\n") : [];

  const { n, eol, L, T, lead, trail, verbatim } = spec;
  const out = new Array(n);
  let rp = 0;
  for (let i = 0; i < n; i++) {
    if (Object.prototype.hasOwnProperty.call(verbatim, i)) {
      out[i] = verbatim[i];
      continue;
    }
    const row = rows[rp++];
    if (row === undefined) throw new Error("ctxfold/logs: missing row " + rp + " (rows fewer than schema declares)");
    const toks = row.length ? row.split(" ") : [];
    if (toks.length < L + T) throw new Error("ctxfold/logs: row " + i + " has " + toks.length + " values, schema needs at least " + (L + T));
    const leadMids = toks.slice(0, L);
    const trailMids = toks.slice(L, L + T);
    const mid = toks.slice(L + T).join(" ");
    const fields = [];
    for (let c = 0; c < L; c++) fields.push(lead[c].pre + leadMids[c] + lead[c].suf);
    if (mid.length) for (const w of mid.split(" ")) fields.push(w);
    for (let c = 0; c < T; c++) fields.push(trail[c].pre + trailMids[c] + trail[c].suf);
    out[i] = fields.join(" ");
  }
  return out.join("\n") + (eol ? "\n" : "");
}

module.exports = { name: "logs", detect, encode, decode };
