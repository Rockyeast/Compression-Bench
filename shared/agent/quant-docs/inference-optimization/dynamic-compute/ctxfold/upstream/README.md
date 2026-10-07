# ctxfold

![ctxfold: raw structured data folds into a compact table the model reads directly](https://raw.githubusercontent.com/antrixy/ctxfold/main/assets/ctxfold-before-after.png)

**Reduce LLM prompt tokens on structured data — losslessly.**

Logs, JSON, and CSV are the bulkiest, most repetitive things we put into prompts.
ctxfold folds their repeated structure into a compact, readable table the model
reads directly — cutting tokens without dropping a byte.

**How:** the same keys, prefixes, and templates repeat on every row, so ctxfold
lifts them into a one-time header and keeps only what varies.

**Why it matters:** an LLM doesn't need the repeated structure — it needs the
information inside it. Strip the repetition, keep the data, and the model answers
exactly as it would from the raw input.

> **Lossless or no-op. Never lossy.** Every encoder ships with a decoder, and
> `compress()` verifies that decoding its output reproduces the input before
> returning it. If it can't, you get your original text back, untouched.


```
   JSON · logs                        the model reads the folded table
   (raw, repetitive)                  directly, answering exactly as it
        │                             would from the raw input
        │  ctxfold.compress()                       ▲
        │  ~39% fewer prompt tokens                 │
        │  (GPT tokenizer; up to ~66% on            │
        │   highly repetitive data)                 │
        ▼                                           │
   ┌──────────────┐ ──────────────────────────────►┘
   │ folded table │
   └──────────────┘
        │
        │  ctxfold.decompress()
        ▼
   original bytes, byte-for-byte   (lossless round-trip, verified before return)
```


CSV folds too (~28% fewer tokens) but the folded form isn't reliably
direct-readable — send CSV raw to the model, or fold it for lossless
pipeline transit only.

**Why not just gzip it?** Because gzip'd JSON isn't model-readable — you'd have to
decompress before the model sees it, saving zero prompt tokens. ctxfold's output
is the rare thing that's *both* readable by the model as-is **and** losslessly
reversible, so the savings land in the prompt where they count.

### The contract

> **ctxfold operates on repeated records.**
> If your data can be viewed as rows with shared structure, ctxfold folds it.
> If it can't, ctxfold intentionally does nothing — it never guesses, and never
> drops data.

It's a data contract, not a prompt trick: either the model receives an
equivalent representation, or the pipeline declines the optimization. The schema
is *derived from the data* at encode time and the round-trip self-check verifies
they agree, so the header can't silently drift from the rows. `validate(payload)`
re-checks that consistency on any folded payload (catches a dropped cell, a
truncated rows section, an out-of-range code) — note it confirms a payload is
sound and decodable, not that it matches an original it never saw.

It isn't a replacement for semantic compression — it's the other half. Summarize
to extract a subset; ctxfold to shrink repetition without losing anything. It
shines on structured data, not prose.

## Benchmarks

Measured with `gpt-4o-mini` and the GPT tokenizer. "Exact match" = field-level
lookups scored against ground truth (folded vs. raw, same questions). The CSV
result was confirmed on `gpt-4o` as well (0/24 on mini).

| Dataset          | Metric             |    Raw | Folded | Result |
| ---------------- | ------------------ | -----: | -----: | ------ |
| JSON (400 recs)  | exact-match lookup |  24/24 |  24/24 | reads identically |
| JSON (400 recs)  | prompt tokens      | 18,052 | 10,956 | **39% fewer** |
| Logs (1,200 ln)  | exact-match lookup |  23/24 |  23/24 | reads identically |
| Logs (1,200 ln)  | prompt tokens      | 45,416 | 27,602 | **39% fewer** |
| CSV / TSV (400r) | exact-match lookup |  24/24 |  ≤9/24 | **not direct-readable** |
| CSV / TSV (400r) | prompt tokens      | 12,042 |  8,686 | 27.9% fewer — pipeline only |

### Cross-model readability

The same harnesses, run unchanged against other providers via their
OpenAI-compatible endpoints. Qwen3.6-27B is a reasoning model and ran with an
enlarged completion budget (`MAX_TOKENS`); dataset sizes vary per run to fit
free-tier rate limits (details below the table).

| Format | gpt-4o-mini | llama-3.3-70b | qwen3.6-27b (reasoning) | claude-haiku-4-5 |
| ------ | ----------- | ------------- | ----------------------- | --------------- |
| JSON   | 24/24, = raw | 24/24, = raw | 24/24, = raw | 24/24, = raw |
| Logs   | folded closer to truth | folded closer to truth | exact on both | folded closer to truth |
| CSV    | fails (1/24) | fails (0/0)¹ | 24/24 via reasoning² | 24/24³ |

Run details:

- **JSON** — exact-match lookup, folded vs raw, scored against ground truth.
  180 records (GPT, Llama, Claude), 80 (Qwen). Every model, both forms: 24/24.
  Claude: chars 24,677→9,747 (60.5%), prompt tokens 9,044→5,848 (35.3% fewer).
- **Logs** — aggregate question (total ERROR lines + top service) on a seeded
  log from `examples/make-sample-log.js`, which prints its exact ground truth.
  On a 260-line log (truth: 23 errors, worker 11), gpt-4o-mini answered
  32/worker-12 on raw vs 22/worker-10 folded; llama-3.3-70b answered
  17/worker-9 raw vs 24/worker-11 folded; claude-haiku-4-5 answered
  20/worker-8 raw vs 21/worker-10 folded — all inexact everywhere, all
  *closer to truth on the folded form*. On a 120-line log (truth: 10 errors,
  api 4), qwen3.6-27b was exactly right on both forms. In no run did folding
  make logs less answerable.
- **CSV** — exact-match lookup through the legend's prefix/suffix rules.
  180 records (GPT, Llama, Claude), 120/40 (Qwen).
  Claude: chars 10,233→8,139 (20.5%), prompt tokens 6,166→4,634 (24.8% fewer).

Notes:

- ¹ Llama's CSV failure mode differs from GPT's: it mangled the folded SKUs
  themselves, so zero answers aligned with any record (0 scoreable fields),
  where GPT aligned records but mis-reconstructed values. Same conclusion,
  worse failure.
- ² The reasoning-model CSV result cuts both ways. Qwen3.6-27B scored 24/24
  in both runs where its reasoning completed (120 and 40 records), visibly
  deriving the legend's prefix/suffix rules step by step — so the fold *is*
  reconstructable by a model, deterministically. But two further runs
  truncated mid-derivation at 3,000 and 4,500 thinking tokens, and at 40
  records the fold only saved ~7% of prompt tokens. Spending thousands of
  variable-length completion tokens to undo a ~15–25% prompt saving is a bad
  trade: the fold doesn't delete the reconstruction work, it moves it into
  the model. CSV therefore stays **pipeline-only** as a recommendation.
- ³ claude-haiku-4-5 read the folded CSV 24/24 with no visible reasoning
  overhead — the only model to do so. One model reading it correctly
  doesn't change the recommendation: folded CSV still fails on GPT and
  Llama, so it's only safe to send directly if you've pinned your model
  *and* measured it, which is the dictionary-coding caveat again. CSV
  stays pipeline-only by default.
- Accuracy is the cross-model claim. Token-% figures elsewhere in this README
  are per-tokenizer (GPT tokenizer unless stated); folded savings measured
  38–46% on Llama's tokenizer, 31–41% on Qwen's, and 24.8–35.3% on Claude's across these runs.

## Install

```bash
npm install ctxfold
```

Zero runtime dependencies. Provider-agnostic — it operates on text, not on an API.

## Quick start

```js
const { compress } = require("ctxfold");

const { text, stats } = compress(bigBlob);
// Send `text` in your prompt instead of the original.
console.log(stats.encoder, `${(stats.tokenRatio * 100).toFixed(0)}% fewer tokens`, "lossless:", stats.lossless);
```

The model reads `text` directly — **there is no decode step at runtime.** The
decoder exists to *prove* losslessness (and is exposed as `decompress(text)` if
you ever want the data back programmatically).

For exact token stats, pass your tokenizer:

```js
const { encode } = require("gpt-tokenizer");
compress(bigBlob, { countTokens: (s) => encode(s).length });
```

## CLI

```bash
cat app.log | ctxfold --stats        # compress stdin -> stdout, stats to stderr
ctxfold data.json > packed.txt        # compress a file
ctxfold --profile data.json           # analyze: where tokens go, what's foldable
ctxfold --dictionary data.json        # opt-in: dictionary-code low-cardinality columns
ctxfold --decompress packed.txt       # reverse it
```

## Token profiler

`--profile` answers "where do this prompt's tokens go, and what would folding
save?" — without transforming anything the model reads.

```
$ ctxfold --profile users.json

[ctxfold profile]
format      JSON array — 300 records × 7 fields
size        65,614 chars ≈ 16,404 tokens (estimated; pass a tokenizer for exact)

where the characters go
  keys         27%   repeated field names (with quotes)
  syntax        7%   braces, brackets, commas, colons
  values       36%   the data itself (with string quotes)
  whitespace   30%   indentation and spacing

foldable (lossless, verified by round-trip)
  fold             -66%   direct-readable — validated 24/24 vs raw
  + --dictionary   -73%   readability tradeoff — off by default, see README

verdict: fold it — 16,404 → ~5,595 tokens (~66% fewer)
```

The numbers follow the same rules as the benchmark table:

- **Composition is measured in characters and attributed exactly** — the
  categories sum to the input size. Attributing individual *tokens* to
  categories would be false precision, so token figures are totals only,
  marked estimated unless you pass a real tokenizer.
- **The "foldable" figures come from actually running `compress()`** on your
  input — the profiler can never promise more than the encoder delivers.
- **Readability claims mirror the measured results**: JSON and logs are
  validated direct-readable; CSV is flagged pipeline-only.

If nothing folds, the profiler says why (quoted CSV, nested JSON, too few
records, prose) instead of silently no-opping.

Programmatic use returns structured data:

```js
const { profile, renderProfile } = require("ctxfold");
const report = profile(text);                 // { format, composition, foldable, verdict, ... }
console.log(renderProfile(report));           // the CLI's text report
```

Try it with zero setup — no API key needed:

```bash
node examples/profile-demo.js
```

## Dictionary coding (opt-in)

For JSON, low-cardinality string columns (a `status`, `category`, or `region`
field with only a handful of distinct values repeated across many rows) can be
**dictionary-coded**: each distinct value becomes a small integer, with the
`code → value` map declared once in the header. It's lossless and pushes JSON
savings from ~39% to ~46% on suitable data.

```js
const { text, stats } = compress(jsonArray, { dictionary: true });
// or: ctxfold --dictionary data.json
```

It's **off by default, on purpose.** In testing, models read the coded columns
slightly less reliably than plain values — they have to resolve `region=1` back
to `EU` through the header, and occasionally don't. So the extra savings come
with a readability cost when the model reads the table *directly*.

Use it when:
- you call `decompress()` to restore real values **before** the model sees them
  (then the codes never reach the model and readability is irrelevant), or
- you've measured that your model + data resolve the dictionary reliably.

Otherwise, leave it off — the default (~39%, fully readable) is the safe choice.
The round-trip self-check still gates it, so it can never be lossy either way.

## CSV folding is pipeline-mode

CSV is the one format where folding is **not** validated for direct model
reading — by measurement, not caution.

The reason is structural. JSON and logs fold by removing *syntax* — keys,
braces, quotes, log templates — and every value the model needs stays verbatim
in the row. CSV has no syntax to remove; it's already a compact table. The only
redundancy left is inside the values, so the CSV encoder affix-factors the data
itself and rows carry only each field's varying middle.

Models don't reliably reconstruct `prefix + middle` at read time. Measured with
the same harness as JSON/logs (`examples/gpt-csv-equivalence.js`): folded CSV
scored 0/24 on gpt-4o-mini and 6–9/24 across runs on gpt-4o against 24/24 raw —
wrong rows matched, prefixes dropped. It's the dictionary-coding tradeoff again, but total:
indirection through a header costs readability, and here the savings *are* the
indirection.

So fold CSV only when the model never sees the folded form — lossless
transit/storage compression in a pipeline that calls `decompress()` before the
prompt is built. For direct model reading, send CSV raw: it's already near its
readable minimum, which is exactly why there was nothing safe to fold.

JSON and logs remain validated for direct reading.

## What it handles

| Input | Detector | Typical token reduction |
|---|---|---|
| Templated logs (timestamp / level / `[scope]` / `key=value`) | newline-delimited, structured lines | ~35–40% |
| JSON array of flat objects — bare `[…]` or wrapped `{"results":[…]}` | top-level array, or an object with one records array | ~39% |
| CSV / TSV (simple, unquoted) | consistent delimiter + header + rows | ~24–28% — **pipeline only**, not direct-readable (see above) |

Anything it doesn't recognize — prose, nested JSON, quoted CSV, an
already-tight table — passes through untouched.

## How it works

One shared primitive does most of the work: **per-column affix factoring**. If
every value in a column shares a prefix or suffix (`reqId=…`, `2026-06-…`,
`USR-…`, `[service]`), that repeated part is written **once** in a header and
each row keeps only what varies. Constant columns collapse to empty cells.
Repeated JSON keys and log templates are lifted into a one-time schema. The
result is a compact, self-labeling table with a `cols:` header the model reads
like a spreadsheet.

Rows that don't fit the dominant template are kept verbatim, so the transform is
always reversible.

## What "lossless" means

- **Logs, CSV/TSV:** byte-for-byte identical after round-trip.
- **JSON:** value-identical — the same data when parsed. (Whitespace and number
  formatting like `1.0` vs `1` are not data; key order is preserved.)

Where the folded form could not express a distinction unambiguously, the record
is kept verbatim rather than flattened — an empty nested object, a value nested
two levels down, or a key holding an object in one record and a scalar in
another. If a literal dotted key would render the same column header as a
flattened path, the whole input declines to flatten instead. Refusing is always
available and always reversible; guessing is not.

Either way, `compress()` checks the round-trip before returning, so you never get
a lossy result.

## Not a semantic compressor

ctxfold doesn't summarize and won't help with prose. Tools that compress
meaning — summarization, retrieval, or token-pruning approaches like
LLMLingua — reach far higher ratios by discarding information they judge
unlikely to matter. That judgment can't be verified task-independently:
whether a dropped detail mattered depends on the question asked later.

ctxfold takes only the compression that costs no trust — structural
redundancy, verified byte-for-byte on every encode. The two compose:
summarize or retrieve to pick what matters, then ctxfold to shrink the
repetition in what's left. Just note the guarantees don't transfer — the
lossless contract covers the fold, not what the picker dropped.

## Tests & benchmark

```bash
npm test                       # lossless round-trips across all three formats
npm i -D gpt-tokenizer         # optional, exact token counts
npm run bench [file]           # token reduction on logs (or your own file)
```

The `examples/` folder has model readability checks: they ask a model to read
records out of the compressed form and score against exact ground truth. They
accept any OpenAI-compatible endpoint (`OPENAI_BASE_URL` + `MODEL` env vars);
results across models are in the cross-model readability table above.

## Scope (v0)

- JSON objects fold **one level deep** — `{"user":{"name":"a"}}` becomes a
  `user.name` column. Deeper nesting, arrays, and empty nested objects are kept
  verbatim rather than flattened lossily.
- CSV/TSV is **simple/unquoted** — quoted fields (commas/newlines inside cells)
  pass through, by design, to guarantee byte-exactness.

## Roadmap

ctxfold's focus is to be the best **tabular** structural folder — not to cover
every format.

**Next up**

- Quoted CSV/TSV support — RFC 4180 field parsing so quoted cells fold; scoped
  to byte-exactness only, since CSV folding is pipeline-mode

**Planned**

- More tabular formats that map cleanly to the same core (SQL result sets,
  Markdown tables, HTML tables)
- Real-world datasets and benchmarks
- Middleware/integrations for common LLM frameworks

**Exploring**

- Dictionary-coding readability — close the gap so `--dictionary` can be safe
  by default
- Readable CSV folding — the current affix mechanism is pipeline-only; a
  direct-readable variant needs a different approach, if one exists
- Python port (`pip install ctxfold`) — open an issue if you want it

**Not in scope**

Hierarchical data (YAML, XML, deeply nested JSON) needs a different algorithm;
if it happens, it'll live as a separate `ctxfold-hierarchical` rather than blur
this one's identity. One level of flattening is the ceiling here — the
column-path encoding could express more, and deliberately does not.

## License

MIT — see [LICENSE](./LICENSE).
