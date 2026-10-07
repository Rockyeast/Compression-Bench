---
layout: default
title: "I built a readability test for my own compression format. It scored 0/24"
permalink: /readability/
---

# I built a readability test for my own compression format. It scored 0/24

*A negative result, its root cause, and the feature it produced.*

When I released [ctxfold](https://github.com/antrixy/ctxfold) — lossless, structure-aware compression for the bulky stuff we put in LLM prompts — its benchmark table had one cell I wasn't proud of:

```plaintext
| CSV / TSV | char reduction | ~30–45%* |

*CSV readability not yet validated against a model
```

JSON and logs had real numbers: a model answering lookup questions off the folded form, scored against exact ground truth, matching raw field-for-field. CSV had a character count and an asterisk.

So I wrote [the same harness](https://github.com/antrixy/ctxfold/blob/main/examples/gpt-csv-equivalence.js) for CSV. Generate 400 records with realistic redundancy, fold them, ask GPT-4o-mini to look up specific records in both forms, score against ground truth.

Raw CSV: **24/24**. Folded CSV: **0/24**.

Not "slightly worse." Zero.

## Maybe the model is too small?

First hypothesis: capability threshold. Re-ran on GPT-4o.

Raw: **24/24**. Folded: **6/24**. A later run: 9/24. So no — a stronger model helps a little and inconsistently. The format is the problem.

The failure had structure, which is what made it diagnosable. Asked about record `NW-1258`, the model reported a warehouse of `WH-2` (a half-applied prefix), a supplier of `DAL-2` (a value from the warehouse column), and a price from some other row entirely. In another run it answered qty `1238` for sku `NW-1238` — it read the record's own identifier back as data. Two distinct failures, every time: it couldn't find the right row, and it couldn't reconstruct the values in it.

## The root cause is embarrassing in hindsight

ctxfold's JSON and logs encoders fold **syntax**. A JSON array of objects repeats every key on every record; the encoder lifts keys, braces, and quotes into a one-time header, and every *value* stays verbatim in its row. Logs are the same story with templates: timestamps, levels, `reqId=` prefixes get lifted; the payload stays put. The model reads a plain labeled table containing exactly the values it needs.

CSV has no syntax to remove. No keys, no braces — it's already a compact table. The only redundancy left is **inside the values**: shared prefixes like `NW-` on every sku, `WH-` on every warehouse, a constant `USD` column. So the CSV encoder factors those out, and each row keeps only the varying middle. Lossless, byte-exact, verified on every encode.

And unreadable. To answer a question, the model has to find a row by a *partial* key (the sku column only contains `258`, because `NW-1` was factored out — all 400 skus shared it) and then reconstruct every value through the header's `prefix + middle` rules. It can't do either reliably. On GPT-4o-mini it essentially can't do it at all.

I'd actually seen a milder version of this before. ctxfold's opt-in dictionary coding (low-cardinality values become small integers, mapped once in the header) pushes JSON savings from ~39% to ~46% — and in testing, models resolved the coded values slightly less reliably than plain ones. That's why it ships off by default. The CSV result is the same phenomenon at full strength:

**Models read values, not reconstruction rules. Indirection through a header costs readability — and in CSV's case, the savings *were* the indirection.**

## What I didn't do

I considered "fixing" the format — don't factor unique columns, keep identifiers whole. But that only patches row-finding; the prefix-reconstruction failure remains, and the savings shrink toward nothing. CSV is already near its readable minimum. That's *why* there was nothing safe to fold.

So the fix was documentation, not code. As of v0.1.4, CSV folding is **pipeline-mode**: fold it for lossless transit or storage, call `decompress()` before the prompt is built, and the model never sees the folded form. For direct model reading, send CSV raw. The benchmark table now says so, with the measured numbers where the asterisk used to be. JSON and logs remain validated direct-readable — their folded output keeps every value intact, and the scores show it.

The rule I keep coming back to: ctxfold's core contract is *lossless or no-op — never lossy*. It turns out the same discipline applies to claims. Either a readability claim has a measurement behind it, or it gets the asterisk. And when the measurement comes in against you, the asterisk becomes documentation.

## The feature the failure produced

Sitting with those numbers, the useful question turned out to be: *before* anyone folds anything — where do a prompt's tokens actually go, and what's safely foldable?

v0.2.0 ships that as `ctxfold --profile`:

```plaintext
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

That's a real API response shape — wrapped array, pretty-printed. 64% of the file is structure and formatting, which is why the fold is fat. On a CSV file, the same command reports the fold as **pipeline-only** and tells you to send it raw or decompress first — the negative result is baked into the tool's own advice.

The profiler follows the rules the failure taught:

- **Composition is measured in characters and attributed exactly** — the categories sum to the input size. Attributing individual *tokens* to categories would be false precision, so token figures are totals, marked estimated unless you pass a real tokenizer.
- **The "foldable" numbers come from actually running the encoder** on your input. The profiler cannot promise more than `compress()` delivers — that's an invariant with a test on it, not a policy.
- **If nothing folds, it says why** — quoted CSV, nested JSON, too few records, prose — instead of silently no-opping.

Try it with zero setup (no API key, deterministic output):

```shell
git clone https://github.com/antrixy/ctxfold && cd ctxfold
node examples/profile-demo.js
```

Or on your own data:

```shell
npm install -g ctxfold
ctxfold --profile your-file.json
```

## The part I'd generalize

If you maintain a tool that makes claims — savings, accuracy, compatibility — the cheapest test you'll ever write is the one that checks your own table. Mine took an afternoon, cost a few cents of API calls, and found that a third of my format list didn't do what a reader would assume. The fix cost nothing but honesty, and the tool that fell out of it is the most useful thing in the package.

The full harness is in the repo (`examples/gpt-csv-equivalence.js`), along with the [v0.1.4](https://github.com/antrixy/ctxfold/releases/tag/v0.1.4) and [v0.2.0](https://github.com/antrixy/ctxfold/releases/tag/v0.2.0) release notes. If you push structured data into prompts and your payloads break my assumptions, I want to hear about it.

---

**Repo & docs:** <https://github.com/antrixy/ctxfold> · **npm:** `npm install ctxfold` · MIT licensed.
