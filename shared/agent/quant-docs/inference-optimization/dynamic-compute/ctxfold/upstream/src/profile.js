"use strict";

// ctxfold profiler: answers "where do this prompt's characters go, and what
// would folding save?" — WITHOUT transforming anything the model reads.
//
// Honesty rules, same as the benchmark table:
//   - Composition is measured in CHARACTERS and attributed exactly: the
//     categories sum to the input size (or the normalized size, see JSON note).
//     Attributing individual TOKENS to categories would be false precision.
//   - Token figures are totals only, and marked estimated unless the caller
//     passes opts.countTokens.
//   - The "foldable" numbers come from actually running compress() on the
//     input — the profiler can never promise more than the encoder delivers.
//   - Readability claims per format mirror the measured results in the README
//     (JSON/logs validated direct-readable; CSV pipeline-only).

// Core is required lazily (inside profile()) because index.js also requires
// this module — a top-level require here would be circular.
const { factorColumn } = require("./affix");

// ---------------------------------------------------------------------------
// JSON composition. Computed over the minified serialization so the category
// sums are exact; whitespace is the difference between the original and the
// minified form (clamped at 0 if number formatting makes it negative).
// String quotes are counted with the item they wrap (keys with keys, values
// with values); syntax is the residue: braces, brackets, commas, colons.

function extractRecordsArray(parsed) {
  if (Array.isArray(parsed)) return parsed;
  if (!parsed || typeof parsed !== "object") return null;
  const keys = Object.keys(parsed);
  const arrayKeys = keys.filter((k) => {
    const v = parsed[k];
    if (!Array.isArray(v) || v.length < 4) return false;
    const objs = v.filter((e) => e && typeof e === "object" && !Array.isArray(e)).length;
    return objs >= v.length * 0.6;
  });
  return arrayKeys.length === 1 ? parsed[arrayKeys[0]] : null;
}

function jsonComposition(text, parsed) {
  const minified = JSON.stringify(parsed);
  let keyChars = 0;
  let valueChars = 0;

  function walk(node) {
    if (Array.isArray(node)) {
      for (const item of node) walk(item);
    } else if (node && typeof node === "object") {
      for (const k of Object.keys(node)) {
        keyChars += k.length + 2; // the key and its quotes
        const v = node[k];
        if (v && typeof v === "object") walk(v);
        else valueChars += JSON.stringify(v).length;
      }
    } else {
      valueChars += JSON.stringify(node).length;
    }
  }
  walk(parsed);

  const syntaxChars = minified.length - keyChars - valueChars;
  const whitespace = Math.max(0, text.length - minified.length);
  const base = keyChars + valueChars + syntaxChars + whitespace;
  return {
    base,
    normalized: text.length < minified.length,
    entries: [
      { label: "keys", chars: keyChars, note: "repeated field names (with quotes)" },
      { label: "syntax", chars: syntaxChars, note: "braces, brackets, commas, colons" },
      { label: "values", chars: valueChars, note: "the data itself (with string quotes)" },
      { label: "whitespace", chars: whitespace, note: "indentation and spacing" },
    ],
  };
}

// ---------------------------------------------------------------------------
// CSV composition. Per-column attribution using the same factorColumn the
// encoder uses, so profile and encoder can't disagree about what's shared.

function pickDelim(lines) {
  for (const d of ["\t", ","]) {
    const counts = lines.map((l) => l.split(d).length);
    const h = counts[0];
    if (h >= 2 && counts.filter((c) => c === h).length / counts.length >= 0.8) return d;
  }
  return null;
}

function csvComposition(text) {
  const eol = text.endsWith("\n");
  const lines = (eol ? text.slice(0, -1) : text).split("\n");
  const delim = pickDelim(lines);
  if (!delim) return null;
  const header = lines[0].split(delim);
  const H = header.length;

  const cols = Array.from({ length: H }, () => []);
  let conforming = 0;
  let verbatimChars = 0;
  let delimChars = 0;
  for (let i = 1; i < lines.length; i++) {
    const f = lines[i].split(delim);
    if (f.length === H) {
      conforming++;
      delimChars += H - 1;
      for (let c = 0; c < H; c++) cols[c].push(f[c]);
    } else {
      verbatimChars += lines[i].length;
    }
  }

  let affixChars = 0;
  let constChars = 0;
  let varyingChars = 0;
  const affixExamples = [];
  const constNames = [];
  for (let c = 0; c < H; c++) {
    const { prefix, suffix, middles } = factorColumn(cols[c]);
    const rows = middles.length;
    if (prefix && middles.every((m) => m === "") && !suffix) {
      constChars += prefix.length * rows;
      constNames.push(header[c]);
    } else {
      affixChars += (prefix.length + suffix.length) * rows;
      for (const m of middles) varyingChars += m.length;
      if (prefix) affixExamples.push(prefix);
      if (suffix) affixExamples.push(suffix);
    }
  }

  // Newlines: one per line except a missing trailing one.
  const newlines = lines.length - (eol ? 0 : 1);
  const headerChars = lines[0].length;
  const base =
    headerChars + affixChars + constChars + varyingChars + delimChars +
    verbatimChars + newlines;

  const entries = [
    { label: "shared affixes", chars: affixChars,
      note: affixExamples.length ? "column prefixes/suffixes (" + affixExamples.slice(0, 3).map((s) => JSON.stringify(s)).join(", ") + ")" : "column prefixes/suffixes" },
    { label: "constant columns", chars: constChars,
      note: constNames.length ? constNames.join(", ") : "none" },
    { label: "varying data", chars: varyingChars, note: "what remains after factoring" },
    { label: "header + delimiters", chars: headerChars + delimChars + newlines, note: "header row, separators, newlines" },
  ];
  if (verbatimChars) entries.push({ label: "non-conforming lines", chars: verbatimChars, note: "kept verbatim" });
  return { base, normalized: false, entries, detail: `${conforming} data rows × ${H} columns` };
}

// ---------------------------------------------------------------------------
// Logs composition. Derived exactly from the fold itself: the rows block of
// the encoded output IS the varying content; everything the fold removed is
// template boilerplate. Two categories, but both are measured, not guessed.

function logsComposition(text, folded) {
  const marker = "\nrows:\n";
  const mi = folded.indexOf(marker);
  if (mi === -1) return null;
  const varying = folded.length - (mi + marker.length);
  const structure = Math.max(0, text.length - varying);
  return {
    base: structure + varying,
    normalized: false,
    entries: [
      { label: "template boilerplate", chars: structure, note: "timestamps, levels, scopes, repeated key= prefixes" },
      { label: "varying content", chars: varying, note: "what remains after folding" },
    ],
  };
}

// ---------------------------------------------------------------------------
// Decline diagnostics for inputs no encoder claims. Heuristic (these re-derive
// common decline conditions rather than instrumenting the encoders), so they
// are phrased as likely reasons, not guarantees.

function declineReasons(text) {
  const reasons = [];
  const t = text.trim();
  const lines = t.split("\n");

  if (t[0] === "[" || t[0] === "{") {
    let parsed = null;
    try { parsed = JSON.parse(t); } catch { reasons.push("looks like JSON but does not parse"); }
    if (parsed) {
      const arr = extractRecordsArray(parsed);
      if (!arr) reasons.push("JSON parses but no single records array was found");
      else if (arr.some((r) => r && typeof r === "object" &&
        Object.values(r).some((v) => v && typeof v === "object"))) {
        reasons.push("JSON records contain nested objects/arrays (flat encoder declines; nesting is on the roadmap)");
      } else if (arr.length < 4) {
        reasons.push("fewer than 4 records — legend overhead would exceed savings");
      }
    }
  }
  if (text.indexOf('"') !== -1) {
    // Quotes break field-count consistency, so don't require pickDelim to
    // succeed — look for a delimiter present on nearly every line instead.
    for (const d of ["\t", ","]) {
      const withDelim = lines.filter((l) => l.indexOf(d) !== -1).length;
      if (lines.length >= 5 && withDelim / lines.length >= 0.8) {
        reasons.push("delimited data with quote characters — quoted CSV passes through by design (lossless-or-no-op)");
        break;
      }
    }
  }
  if (lines.length < 5) reasons.push("fewer than 5 lines — too small to fold profitably");
  if (reasons.length === 0) reasons.push("no repeated record structure detected (prose and free text pass through by design)");
  return reasons;
}

// ---------------------------------------------------------------------------

const READABILITY = {
  json: "direct-readable — validated 24/24 vs raw",
  logs: "direct-readable — validated 23/24 vs raw",
  csv: "pipeline-only — NOT direct-readable (measured 0\u20139/24)",
};

function profile(text, opts = {}) {
  const { compress, estimateTokens } = require("./index");
  if (typeof text !== "string") throw new TypeError("profile(text): text must be a string");
  const count = opts.countTokens || estimateTokens;
  const tokensExact = !!opts.countTokens;

  const { text: folded, stats } = compress(text, opts);
  const out = {
    format: stats.encoder,
    detail: "",
    chars: text.length,
    tokens: count(text),
    tokensExact,
    composition: null,
    compositionNormalized: false,
    foldable: [],
    verdict: "",
    reasons: null,
  };

  if (stats.encoder === "none") {
    out.reasons = declineReasons(text);
    out.verdict = "nothing to fold — ctxfold declines rather than guesses (by design)";
    return out;
  }

  // Composition per format.
  let comp = null;
  if (stats.encoder === "json") {
    const parsed = JSON.parse(text.trim());
    const arr = extractRecordsArray(parsed);
    comp = jsonComposition(text, parsed);
    const fields = arr && arr.length ? Object.keys(arr[0]).length : 0;
    out.detail = `JSON array — ${arr ? arr.length : "?"} records × ${fields} fields`;
  } else if (stats.encoder === "csv") {
    comp = csvComposition(text);
    out.detail = comp && comp.detail ? `CSV/TSV — ${comp.detail}` : "CSV/TSV";
  } else if (stats.encoder === "logs") {
    comp = logsComposition(text, folded);
    const n = text.split("\n").filter((l) => l.length > 0).length;
    out.detail = `templated logs — ${n} lines`;
  }
  if (comp) {
    out.compositionNormalized = comp.normalized;
    out.composition = comp.entries.map((e) => ({
      label: e.label, chars: e.chars,
      pct: comp.base ? e.chars / comp.base : 0,
      note: e.note,
    }));
  }

  // Foldable: only what compress() actually achieved on this input.
  out.foldable.push({
    label: "fold",
    tokenRatio: stats.tokenRatio,
    tokensAfter: stats.tokensAfter,
    note: READABILITY[stats.encoder],
  });
  if (stats.encoder === "json" && !opts.dictionary) {
    const dict = compress(text, Object.assign({}, opts, { dictionary: true }));
    if (dict.stats.encoder === "json" && dict.stats.tokensAfter < stats.tokensAfter) {
      out.foldable.push({
        label: "+ --dictionary",
        tokenRatio: dict.stats.tokenRatio,
        tokensAfter: dict.stats.tokensAfter,
        note: "readability tradeoff — off by default, see README",
      });
    }
  }

  // Verdict.
  const pct = Math.round(stats.tokenRatio * 100);
  if (stats.encoder === "csv") {
    out.verdict = "already near its readable minimum — fold only if you decompress() " +
      "before the model reads it; otherwise send raw";
  } else if (stats.tokenRatio < 0.1) {
    out.verdict = `marginal — folding saves ~${pct}%; worth it only if tokens are expensive to you`;
  } else {
    out.verdict = `fold it — ${out.tokens.toLocaleString("en-US")} \u2192 ~${stats.tokensAfter.toLocaleString("en-US")} tokens (~${pct}% fewer)`;
  }
  return out;
}

// Text renderer used by the CLI (and reusable by anything else that wants the
// human-readable report).
function renderProfile(p) {
  const L = [];
  L.push("[ctxfold profile]");
  L.push(`format      ${p.format === "none" ? "unrecognized" : p.detail || p.format}`);
  L.push(`size        ${p.chars.toLocaleString("en-US")} chars ${p.tokensExact ? "=" : "\u2248"} ${p.tokens.toLocaleString("en-US")} tokens${p.tokensExact ? "" : " (estimated; pass a tokenizer for exact)"}`);

  if (p.composition) {
    L.push("");
    L.push(p.compositionNormalized
      ? "where the characters go (normalized form)"
      : "where the characters go");
    const width = Math.max(...p.composition.map((c) => c.label.length));
    for (const c of p.composition) {
      const pctStr = (c.pct * 100).toFixed(0).padStart(3) + "%";
      L.push(`  ${c.label.padEnd(width)}  ${pctStr}   ${c.note}`);
    }
  }

  if (p.foldable.length) {
    L.push("");
    L.push("foldable (lossless, verified by round-trip)");
    const width = Math.max(...p.foldable.map((f) => f.label.length));
    for (const f of p.foldable) {
      const pctStr = ("-" + Math.round(f.tokenRatio * 100) + "%").padStart(5);
      L.push(`  ${f.label.padEnd(width)}  ${pctStr}   ${f.note}`);
    }
  }

  if (p.reasons) {
    L.push("");
    L.push("why nothing folded");
    for (const r of p.reasons) L.push(`  - ${r}`);
  }

  L.push("");
  L.push(`verdict: ${p.verdict}`);
  return L.join("\n") + "\n";
}

module.exports = { profile, renderProfile };
