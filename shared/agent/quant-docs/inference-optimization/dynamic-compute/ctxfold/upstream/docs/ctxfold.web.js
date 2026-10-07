/* ctxfold 0.3.0 — browser bundle. GENERATED from src/ by esbuild; do not edit by hand. */
"use strict";
var ctxfold = (() => {
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __commonJS = (cb, mod) => function __require() {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  };

  // src/affix.js
  var require_affix = __commonJS({
    "src/affix.js"(exports, module) {
      "use strict";
      function longestCommonPrefix(strings) {
        if (strings.length === 0) return "";
        let prefix = strings[0];
        for (let i = 1; i < strings.length && prefix; i++) {
          const s = strings[i];
          let j = 0;
          const max = Math.min(prefix.length, s.length);
          while (j < max && prefix[j] === s[j]) j++;
          prefix = prefix.slice(0, j);
        }
        return prefix;
      }
      function longestCommonSuffix(strings) {
        if (strings.length === 0) return "";
        let suffix = strings[0];
        for (let i = 1; i < strings.length && suffix; i++) {
          const s = strings[i];
          let j = 0;
          const max = Math.min(suffix.length, s.length);
          while (j < max && suffix[suffix.length - 1 - j] === s[s.length - 1 - j]) j++;
          suffix = suffix.slice(suffix.length - j);
        }
        return suffix;
      }
      function factorColumn(values) {
        if (values.length === 0) return { prefix: "", suffix: "", middles: [] };
        let prefix = longestCommonPrefix(values);
        const afterPrefix = values.map((v) => v.slice(prefix.length));
        let suffix = longestCommonSuffix(afterPrefix);
        const middles = afterPrefix.map((v) => v.slice(0, v.length - suffix.length));
        for (let i = 0; i < values.length; i++) {
          if (prefix + middles[i] + suffix !== values[i]) {
            return { prefix: "", suffix: "", middles: values.slice() };
          }
        }
        return { prefix, suffix, middles };
      }
      module.exports = { longestCommonPrefix, longestCommonSuffix, factorColumn };
    }
  });

  // src/encoders/logs.js
  var require_logs = __commonJS({
    "src/encoders/logs.js"(exports, module) {
      "use strict";
      var { factorColumn } = require_affix();
      var RE_TS = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?$/;
      var LEVELS = /* @__PURE__ */ new Set(["TRACE", "DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL"]);
      var RE_KV = /^([A-Za-z_][\w.\-]*)=/;
      var RE_BRACKET = /^\[.*\]$/;
      var RE_NUM = /^-?\d+(?:\.\d+)?$/;
      function fieldType(f) {
        if (RE_TS.test(f)) return "ts";
        if (LEVELS.has(f)) return "lvl";
        if (RE_BRACKET.test(f)) return "br";
        if (RE_KV.test(f)) return "kv";
        if (RE_NUM.test(f)) return "num";
        return "w";
      }
      function leadLen(fields) {
        let i = 0;
        while (i < fields.length) {
          const t = fieldType(fields[i]);
          if (t === "ts" || t === "lvl" || t === "br" || t === "kv") i++;
          else break;
        }
        return i;
      }
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
        const counts = /* @__PURE__ */ new Map();
        for (const p of pairs) counts.set(p, (counts.get(p) || 0) + 1);
        let best = null, bestN = -1;
        for (const [p, n] of counts) if (n > bestN) {
          best = p;
          bestN = n;
        }
        return best;
      }
      function encode(text) {
        const eol = text.endsWith("\n");
        const raw = text.split("\n");
        const lines = eol ? raw.slice(0, -1) : raw;
        const n = lines.length;
        const sigs = [];
        for (const l of lines) {
          const f = l.split(" ");
          const L2 = leadLen(f);
          const T2 = trailLen(f, L2);
          sigs.push(`${L2},${T2}`);
        }
        const [L, T] = modal(sigs).split(",").map(Number);
        if (L + T === 0) return { ok: false };
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
        const used = /* @__PURE__ */ new Map();
        const nameOf = (values, idx) => {
          const v = values[0] || "";
          const t = fieldType(v);
          let base;
          if (t === "kv") {
            const m = v.match(RE_KV);
            base = m ? m[1] : "kv";
          } else if (t === "ts") base = "time";
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
          v: 1,
          n,
          eol,
          L,
          T,
          lead: leadF.map((c) => ({ pre: c.prefix, suf: c.suffix })),
          trail: trailF.map((c) => ({ pre: c.prefix, suf: c.suffix })),
          verbatim
        };
        const rows = [];
        for (let r = 0; r < conformingIdx.length; r++) {
          const cells = [];
          for (let c = 0; c < L; c++) cells.push(leadF[c].middles[r]);
          for (let c = 0; c < T; c++) cells.push(trailF[c].middles[r]);
          const mid = middles[r];
          rows.push(mid.length ? cells.join(" ") + " " + mid : cells.join(" "));
        }
        const legend = `legend: each row is space-separated as: ${colNames.slice(0, -1).join(" ")} then the message (everything after the first ${L + T} values).`;
        const out = "\u27E6cf/logs v1\u27E7\nspec " + JSON.stringify(spec) + "\n" + legend + "\ncols: " + colNames.join(" ") + "\nrows:\n" + rows.join("\n");
        return { ok: true, encoded: out };
      }
      function decode(encoded) {
        const nl = encoded.indexOf("\n");
        const magic = encoded.slice(0, nl);
        if (magic !== "\u27E6cf/logs v1\u27E7") throw new Error("bad magic");
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
          if (row === void 0) throw new Error("ctxfold/logs: missing row " + rp + " (rows fewer than schema declares)");
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
    }
  });

  // src/encoders/json.js
  var require_json = __commonJS({
    "src/encoders/json.js"(exports, module) {
      "use strict";
      function scalarKind(v) {
        if (v === null) return "null";
        const t = typeof v;
        if (t === "string" || t === "number" || t === "boolean") return t;
        return "complex";
      }
      function isPlainObject(v) {
        return v !== null && typeof v === "object" && !Array.isArray(v);
      }
      function detect(text) {
        const t = text.trim();
        if (t.length <= 200 || !/\[\s*\{/.test(t)) return false;
        return t[0] === "[" || t[0] === "{";
      }
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
        if (arrayKeys.length !== 1) return null;
        const arrayKey = arrayKeys[0];
        const rest = {};
        for (const k of order) if (k !== arrayKey) rest[k] = parsed[k];
        return { arr: parsed[arrayKey], wrap: { order, arrayKey, rest } };
      }
      function recordPaths(obj, flatten) {
        if (!isPlainObject(obj)) return null;
        const paths = [];
        for (const k of Object.keys(obj)) {
          const v = obj[k];
          if (scalarKind(v) !== "complex") {
            paths.push([k]);
            continue;
          }
          if (!flatten) return null;
          if (!isPlainObject(v)) return null;
          const inner = Object.keys(v);
          if (inner.length === 0) return null;
          for (const ik of inner) {
            if (scalarKind(v[ik]) === "complex") return null;
            paths.push([k, ik]);
          }
        }
        return paths;
      }
      function valueAt(obj, path) {
        return path.length === 1 ? obj[path[0]] : obj[path[0]][path[1]];
      }
      function encodeCell(v) {
        switch (scalarKind(v)) {
          case "string":
            return "s" + JSON.stringify(v);
          case "number":
            return "n" + String(v);
          case "boolean":
            return v ? "b1" : "b0";
          case "null":
            return "z";
          default:
            return null;
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
      function columnType(values) {
        const kinds = new Set(values.map(scalarKind));
        if (kinds.size === 1) {
          const only = kinds.values().next().value;
          if (only === "string") {
            if (values.every((v) => v.indexOf("	") === -1 && v.indexOf("\n") === -1)) return "s";
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
          case "s":
            return v;
          case "n":
            return String(v);
          case "b":
            return v ? "1" : "0";
          case "z":
            return "";
          default:
            return encodeCell(v);
        }
      }
      function decodeByType(cell, t) {
        switch (t) {
          case "s":
            return cell;
          case "n":
            return Number(cell);
          case "b":
            return cell === "1";
          case "z":
            return null;
          default:
            return decodeCell(cell);
        }
      }
      function buildDictionary(values, { maxDistinct = 256 } = {}) {
        const index = /* @__PURE__ */ new Map();
        const dict = [];
        for (const v of values) {
          if (!index.has(v)) {
            index.set(v, dict.length);
            dict.push(v);
          }
        }
        const d = dict.length;
        if (d < 2 || d > maxDistinct || d >= values.length) return null;
        const avgVal = dict.reduce((s, v) => s + v.length, 0) / d;
        const avgCode = values.reduce((s, v) => s + String(index.get(v)).length, 0) / values.length;
        const rowSavings = values.length * (avgVal - avgCode);
        const headerCost = dict.reduce((s, v, i) => s + String(i).length + 1 + v.length + 2, 0) + 16;
        if (rowSavings - headerCost <= 0) return null;
        return { dict, index };
      }
      function layout(arr, flatten) {
        const sigCount = /* @__PURE__ */ new Map(), sigPaths = /* @__PURE__ */ new Map(), perRecord = new Array(arr.length);
        for (let i = 0; i < arr.length; i++) {
          const paths2 = recordPaths(arr[i], flatten);
          perRecord[i] = paths2;
          if (!paths2) continue;
          const sig = JSON.stringify(paths2);
          sigCount.set(sig, (sigCount.get(sig) || 0) + 1);
          if (!sigPaths.has(sig)) sigPaths.set(sig, paths2);
        }
        if (sigCount.size === 0) return null;
        let bestSig = null, bestN = -1;
        for (const [sig, n] of sigCount) if (n > bestN) {
          bestSig = sig;
          bestN = n;
        }
        if (bestN < 4) return null;
        const paths = sigPaths.get(bestSig);
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
        try {
          parsed = JSON.parse(text);
        } catch {
          return { ok: false };
        }
        const extracted = extractArray(parsed);
        if (!extracted) return { ok: false };
        const { arr, wrap } = extracted;
        if (!Array.isArray(arr) || arr.length < 4) return { ok: false };
        const plan = layout(arr, true) || layout(arr, false);
        if (!plan) return { ok: false };
        const { paths, conforming, verbatim } = plan;
        const names = paths.map((p) => p.join("."));
        const plans = paths.map((p) => {
          const vals = conforming.map((i) => valueAt(arr[i], p));
          const t = columnType(vals);
          if (opts.dictionary && t === "s") {
            const dic = buildDictionary(vals);
            if (dic) return { t: "d", dict: dic.dict, index: dic.index };
          }
          return { t };
        });
        const rows = conforming.map(
          (i) => paths.map((p, c) => {
            const pl = plans[c];
            return pl.t === "d" ? String(pl.index.get(valueAt(arr[i], p))) : encodeByType(valueAt(arr[i], p), pl.t);
          }).join("	")
        );
        const nested = paths.some((p) => p.length > 1);
        const cols = paths.map((p, c) => {
          const col = { k: names[c] };
          if (p.length > 1) col.p = p;
          if (plans[c].t === "d") {
            col.t = "d";
            col.dict = plans[c].dict;
          } else col.t = plans[c].t;
          return col;
        });
        const spec = { v: nested ? 2 : 1, n: arr.length, cols, verbatim, wrap };
        const dictNote = plans.some((pl) => pl.t === "d") ? " Dictionary columns (code=value): " + names.map((k, c) => plans[c].t === "d" ? `${k} {${plans[c].dict.map((v, idx) => `${idx}=${v}`).join(", ")}}` : null).filter(Boolean).join("; ") + "." : "";
        const nestNote = nested ? " Nested fields, written parent.child: " + names.filter((k, c) => paths[c].length > 1).join(", ") + "." : "";
        const wrapNote = wrap ? `a JSON object with keys [${wrap.order.join(", ")}], where "${wrap.arrayKey}" is the array below` : `a JSON array`;
        const legend = `legend: ${wrapNote} of ${arr.length} objects as a table. Columns: ${names.join(", ")}. Each row is tab-separated values in that order, in array order. Values are bare unless noted.${nestNote}${dictNote}`;
        const out = "\u27E6cf/json v1\u27E7\nspec " + JSON.stringify(spec) + "\n" + legend + "\ncols: " + names.join("	") + "\nrows:\n" + rows.join("\n");
        return { ok: true, encoded: out };
      }
      var MAX_SPEC_VERSION = 2;
      function decode(encoded) {
        const nl = encoded.indexOf("\n");
        if (encoded.slice(0, nl) !== "\u27E6cf/json v1\u27E7") throw new Error("bad magic");
        let rest = encoded.slice(nl + 1);
        const specEnd = rest.indexOf("\n");
        const spec = JSON.parse(rest.slice("spec ".length, specEnd));
        rest = rest.slice(specEnd + 1);
        if (!Number.isInteger(spec.v) || spec.v < 1 || spec.v > MAX_SPEC_VERSION) {
          throw new Error(
            "ctxfold/json: unsupported spec version " + spec.v + " (this build reads up to " + MAX_SPEC_VERSION + ")"
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
          if (line === void 0) throw new Error("ctxfold/json: missing row (rows fewer than schema declares)");
          const cells = line.split("	");
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
            const path = col.p || [col.k];
            if (path.length === 1) {
              obj[path[0]] = value;
            } else {
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
      function verify(original, restored) {
        try {
          return JSON.stringify(JSON.parse(original)) === restored;
        } catch {
          return false;
        }
      }
      module.exports = { name: "json", detect, encode, decode, verify };
    }
  });

  // src/encoders/csv.js
  var require_csv = __commonJS({
    "src/encoders/csv.js"(exports, module) {
      "use strict";
      var { factorColumn } = require_affix();
      function pickDelim(lines) {
        for (const d of ["	", ","]) {
          const counts = lines.map((l) => l.split(d).length);
          const h = counts[0];
          if (h >= 2 && counts.filter((c) => c === h).length / counts.length >= 0.8) return d;
        }
        return null;
      }
      function detect(text) {
        if (text.indexOf('"') !== -1) return false;
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
          v: 1,
          n,
          eol,
          delim,
          header,
          cols: factored.map((c) => ({ pre: c.prefix, suf: c.suffix })),
          verbatim
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
        const legend = `legend: ${delim === "	" ? "TSV" : "CSV"} with header + ${conformingIdx.length} rows. Each row lists values for: ${header.join(", ")} (original delimiter). ` + (notes.length ? `Factored columns: ${notes.join("; ")}.` : "");
        const out = "\u27E6cf/csv v1\u27E7\nspec " + JSON.stringify(spec) + "\n" + legend + "\ncols: " + header.join(delim) + "\nrows:\n" + rows.join("\n");
        return { ok: true, encoded: out };
      }
      function decode(encoded) {
        const nl = encoded.indexOf("\n");
        if (encoded.slice(0, nl) !== "\u27E6cf/csv v1\u27E7") throw new Error("bad magic");
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
          if (rowLine === void 0) throw new Error("ctxfold/csv: missing row (rows fewer than schema declares)");
          const mids = rowLine.split(delim);
          if (mids.length !== cols.length) throw new Error("ctxfold/csv: row has " + mids.length + " fields, schema declares " + cols.length + " columns");
          const fields = cols.map((c, idx) => c.pre + mids[idx] + c.suf);
          out[i] = fields.join(delim);
        }
        return out.join("\n") + (eol ? "\n" : "");
      }
      module.exports = { name: "csv", detect, encode, decode };
    }
  });

  // src/profile.js
  var require_profile = __commonJS({
    "src/profile.js"(exports, module) {
      "use strict";
      var { factorColumn } = require_affix();
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
              keyChars += k.length + 2;
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
            { label: "whitespace", chars: whitespace, note: "indentation and spacing" }
          ]
        };
      }
      function pickDelim(lines) {
        for (const d of ["	", ","]) {
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
        const newlines = lines.length - (eol ? 0 : 1);
        const headerChars = lines[0].length;
        const base = headerChars + affixChars + constChars + varyingChars + delimChars + verbatimChars + newlines;
        const entries = [
          {
            label: "shared affixes",
            chars: affixChars,
            note: affixExamples.length ? "column prefixes/suffixes (" + affixExamples.slice(0, 3).map((s) => JSON.stringify(s)).join(", ") + ")" : "column prefixes/suffixes"
          },
          {
            label: "constant columns",
            chars: constChars,
            note: constNames.length ? constNames.join(", ") : "none"
          },
          { label: "varying data", chars: varyingChars, note: "what remains after factoring" },
          { label: "header + delimiters", chars: headerChars + delimChars + newlines, note: "header row, separators, newlines" }
        ];
        if (verbatimChars) entries.push({ label: "non-conforming lines", chars: verbatimChars, note: "kept verbatim" });
        return { base, normalized: false, entries, detail: `${conforming} data rows \xD7 ${H} columns` };
      }
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
            { label: "varying content", chars: varying, note: "what remains after folding" }
          ]
        };
      }
      function declineReasons(text) {
        const reasons = [];
        const t = text.trim();
        const lines = t.split("\n");
        if (t[0] === "[" || t[0] === "{") {
          let parsed = null;
          try {
            parsed = JSON.parse(t);
          } catch {
            reasons.push("looks like JSON but does not parse");
          }
          if (parsed) {
            const arr = extractRecordsArray(parsed);
            if (!arr) reasons.push("JSON parses but no single records array was found");
            else if (arr.some((r) => r && typeof r === "object" && Object.values(r).some((v) => v && typeof v === "object"))) {
              reasons.push("JSON records contain nested objects/arrays (flat encoder declines; nesting is on the roadmap)");
            } else if (arr.length < 4) {
              reasons.push("fewer than 4 records \u2014 legend overhead would exceed savings");
            }
          }
        }
        if (text.indexOf('"') !== -1) {
          for (const d of ["	", ","]) {
            const withDelim = lines.filter((l) => l.indexOf(d) !== -1).length;
            if (lines.length >= 5 && withDelim / lines.length >= 0.8) {
              reasons.push("delimited data with quote characters \u2014 quoted CSV passes through by design (lossless-or-no-op)");
              break;
            }
          }
        }
        if (lines.length < 5) reasons.push("fewer than 5 lines \u2014 too small to fold profitably");
        if (reasons.length === 0) reasons.push("no repeated record structure detected (prose and free text pass through by design)");
        return reasons;
      }
      var READABILITY = {
        json: "direct-readable \u2014 validated 24/24 vs raw",
        logs: "direct-readable \u2014 validated 23/24 vs raw",
        csv: "pipeline-only \u2014 NOT direct-readable (measured 0\u20139/24)"
      };
      function profile(text, opts = {}) {
        const { compress, estimateTokens } = require_src();
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
          reasons: null
        };
        if (stats.encoder === "none") {
          out.reasons = declineReasons(text);
          out.verdict = "nothing to fold \u2014 ctxfold declines rather than guesses (by design)";
          return out;
        }
        let comp = null;
        if (stats.encoder === "json") {
          const parsed = JSON.parse(text.trim());
          const arr = extractRecordsArray(parsed);
          comp = jsonComposition(text, parsed);
          const fields = arr && arr.length ? Object.keys(arr[0]).length : 0;
          out.detail = `JSON array \u2014 ${arr ? arr.length : "?"} records \xD7 ${fields} fields`;
        } else if (stats.encoder === "csv") {
          comp = csvComposition(text);
          out.detail = comp && comp.detail ? `CSV/TSV \u2014 ${comp.detail}` : "CSV/TSV";
        } else if (stats.encoder === "logs") {
          comp = logsComposition(text, folded);
          const n = text.split("\n").filter((l) => l.length > 0).length;
          out.detail = `templated logs \u2014 ${n} lines`;
        }
        if (comp) {
          out.compositionNormalized = comp.normalized;
          out.composition = comp.entries.map((e) => ({
            label: e.label,
            chars: e.chars,
            pct: comp.base ? e.chars / comp.base : 0,
            note: e.note
          }));
        }
        out.foldable.push({
          label: "fold",
          tokenRatio: stats.tokenRatio,
          tokensAfter: stats.tokensAfter,
          note: READABILITY[stats.encoder]
        });
        if (stats.encoder === "json" && !opts.dictionary) {
          const dict = compress(text, Object.assign({}, opts, { dictionary: true }));
          if (dict.stats.encoder === "json" && dict.stats.tokensAfter < stats.tokensAfter) {
            out.foldable.push({
              label: "+ --dictionary",
              tokenRatio: dict.stats.tokenRatio,
              tokensAfter: dict.stats.tokensAfter,
              note: "readability tradeoff \u2014 off by default, see README"
            });
          }
        }
        const pct = Math.round(stats.tokenRatio * 100);
        if (stats.encoder === "csv") {
          out.verdict = "already near its readable minimum \u2014 fold only if you decompress() before the model reads it; otherwise send raw";
        } else if (stats.tokenRatio < 0.1) {
          out.verdict = `marginal \u2014 folding saves ~${pct}%; worth it only if tokens are expensive to you`;
        } else {
          out.verdict = `fold it \u2014 ${out.tokens.toLocaleString("en-US")} \u2192 ~${stats.tokensAfter.toLocaleString("en-US")} tokens (~${pct}% fewer)`;
        }
        return out;
      }
      function renderProfile(p) {
        const L = [];
        L.push("[ctxfold profile]");
        L.push(`format      ${p.format === "none" ? "unrecognized" : p.detail || p.format}`);
        L.push(`size        ${p.chars.toLocaleString("en-US")} chars ${p.tokensExact ? "=" : "\u2248"} ${p.tokens.toLocaleString("en-US")} tokens${p.tokensExact ? "" : " (estimated; pass a tokenizer for exact)"}`);
        if (p.composition) {
          L.push("");
          L.push(p.compositionNormalized ? "where the characters go (normalized form)" : "where the characters go");
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
    }
  });

  // src/index.js
  var require_src = __commonJS({
    "src/index.js"(exports, module) {
      var logs = require_logs();
      var json = require_json();
      var csv = require_csv();
      var ENCODERS = [json, csv, logs];
      function verifyDefault(original, restored) {
        return original === restored;
      }
      function estimateTokens(s) {
        return Math.ceil(s.length / 4);
      }
      function compress(text, opts = {}) {
        if (typeof text !== "string") throw new TypeError("compress(text): text must be a string");
        const count = opts.countTokens || estimateTokens;
        const only = opts.only ? new Set([].concat(opts.only)) : null;
        const base = {
          encoder: "none",
          lossless: true,
          charsBefore: text.length,
          charsAfter: text.length,
          tokensBefore: count(text),
          tokensAfter: count(text)
        };
        for (const enc of ENCODERS) {
          if (only && !only.has(enc.name)) continue;
          let result;
          try {
            if (!enc.detect(text)) continue;
            result = enc.encode(text, opts);
          } catch {
            continue;
          }
          if (!result || !result.ok) continue;
          let restored;
          try {
            restored = enc.decode(result.encoded);
          } catch {
            continue;
          }
          const verify = enc.verify || verifyDefault;
          if (!verify(text, restored)) continue;
          if (result.encoded.length >= text.length) continue;
          return {
            text: result.encoded,
            stats: {
              encoder: enc.name,
              lossless: true,
              charsBefore: text.length,
              charsAfter: result.encoded.length,
              charRatio: 1 - result.encoded.length / text.length,
              tokensBefore: count(text),
              tokensAfter: count(result.encoded),
              tokenRatio: 1 - count(result.encoded) / count(text)
            }
          };
        }
        return { text, stats: base };
      }
      function decompress(encoded) {
        if (typeof encoded !== "string") return encoded;
        const enc = ENCODERS.find((e) => encoded.startsWith("\u27E6cf/" + e.name));
        if (!enc) return encoded;
        return enc.decode(encoded);
      }
      function validate(payload) {
        if (typeof payload !== "string") return { valid: false, reason: "input is not a string" };
        const enc = ENCODERS.find((e) => payload.startsWith("\u27E6cf/" + e.name));
        if (!enc) return { valid: false, reason: "not a ctxfold payload (no cf/ header)" };
        try {
          enc.decode(payload);
          return { valid: true, encoder: enc.name };
        } catch (e) {
          return { valid: false, encoder: enc.name, reason: e.message };
        }
      }
      module.exports = { compress, decompress, validate, estimateTokens, ENCODERS };
      var { profile, renderProfile } = require_profile();
      module.exports.profile = profile;
      module.exports.renderProfile = renderProfile;
    }
  });
  return require_src();
})();
