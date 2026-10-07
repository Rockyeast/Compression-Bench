"use strict";

const { factorColumn } = require("../affix");

// CSV/TSV encoder. For SIMPLE (unquoted) comma- or tab-delimited tables, factors
// each column's shared prefix/suffix and collapses constant columns to empty
// cells — losslessly, byte-for-byte. Anything with quoted fields (commas or
// newlines hidden inside cells) is declined, because quote-parsing risks
// corruption and our rule is lossless-or-no-op.

function pickDelim(lines) {
  for (const d of ["\t", ","]) {
    const counts = lines.map((l) => l.split(d).length);
    const h = counts[0];
    if (h >= 2 && counts.filter((c) => c === h).length / counts.length >= 0.8) return d;
  }
  return null;
}

function detect(text) {
  if (text.indexOf('"') !== -1) return false; // simple/unquoted only
  const eol = text.endsWith("\n");
  const lines = (eol ? text.slice(0, -1) : text).split("\n");
  if (lines.length < 5) return false;
  return pickDelim(lines) !== null;
}

function encode(text) {
  if (text.indexOf('"') !== -1) return { ok: false };
  const eol = text.endsWith("\n");
  const lines = (eol ? text.slice(0, -1) : text).split("\n");
  const n = lines.length;
  if (n < 5) return { ok: false };

  const delim = pickDelim(lines);
  if (!delim) return { ok: false };

  const header = lines[0].split(delim);
  const H = header.length;
  if (H < 2) return { ok: false };

  // Data rows (index >= 1) with the right column count conform; others verbatim.
  const conformingIdx = [];
  const verbatim = {};
  const cols = Array.from({ length: H }, () => []);
  for (let i = 1; i < n; i++) {
    const f = lines[i].split(delim);
    if (f.length === H) {
      conformingIdx.push(i);
      for (let c = 0; c < H; c++) cols[c].push(f[c]);
    } else {
      verbatim[i] = lines[i];
    }
  }
  if (conformingIdx.length < 4) return { ok: false };

  const factored = cols.map(factorColumn);

  const spec = {
    v: 1, n, eol, delim,
    header,
    cols: factored.map((c) => ({ pre: c.prefix, suf: c.suffix })),
    verbatim,
  };

  const rows = [];
  for (let r = 0; r < conformingIdx.length; r++) {
    rows.push(factored.map((c) => c.middles[r]).join(delim));
  }

  const notes = [];
  for (let c = 0; c < H; c++) {
    const { prefix, suffix } = factored[c];
    if (prefix && factored[c].middles.every((m) => m === "")) notes.push(`${header[c]}=const "${prefix}"`);
    else if (prefix || suffix) notes.push(`${header[c]}="${prefix}"+value+"${suffix}"`);
  }
  const legend =
    `legend: ${delim === "\t" ? "TSV" : "CSV"} with header + ${conformingIdx.length} rows. ` +
    `Each row lists values for: ${header.join(", ")} (original delimiter). ` +
    (notes.length ? `Factored columns: ${notes.join("; ")}.` : "");

  const out =
    "\u27e6cf/csv v1\u27e7\n" +
    "spec " + JSON.stringify(spec) + "\n" +
    legend + "\n" +
    "cols: " + header.join(delim) + "\n" +
    "rows:\n" +
    rows.join("\n");

  return { ok: true, encoded: out };
}

function decode(encoded) {
  const nl = encoded.indexOf("\n");
  if (encoded.slice(0, nl) !== "\u27e6cf/csv v1\u27e7") throw new Error("bad magic");
  let rest = encoded.slice(nl + 1);

  const specEnd = rest.indexOf("\n");
  const spec = JSON.parse(rest.slice("spec ".length, specEnd));
  rest = rest.slice(specEnd + 1);

  const marker = "\nrows:\n";
  const rowsBlock = rest.slice(rest.indexOf(marker) + marker.length);
  const rowLines = rowsBlock.length ? rowsBlock.split("\n") : [];

  const { n, eol, delim, header, cols, verbatim } = spec;
  const out = new Array(n);
  out[0] = header.join(delim);
  let rp = 0;
  for (let i = 1; i < n; i++) {
    if (Object.prototype.hasOwnProperty.call(verbatim, i)) {
      out[i] = verbatim[i];
      continue;
    }
    const rowLine = rowLines[rp++];
    if (rowLine === undefined) throw new Error("ctxfold/csv: missing row (rows fewer than schema declares)");
    const mids = rowLine.split(delim);
    if (mids.length !== cols.length) throw new Error("ctxfold/csv: row has " + mids.length + " fields, schema declares " + cols.length + " columns");
    const fields = cols.map((c, idx) => c.pre + mids[idx] + c.suf);
    out[i] = fields.join(delim);
  }
  return out.join("\n") + (eol ? "\n" : "");
}

module.exports = { name: "csv", detect, encode, decode };
