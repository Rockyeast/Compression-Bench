# PackRat v2 — Full Test Report

**Date:** 2026-04-10
**Branch:** `v2-token-optimization`
**Tester:** Claude Opus 4.6 (automated)

## Executive Summary

| Category | Tests | Pass | Fail | Status |
|----------|-------|------|------|--------|
| Edge Cases (unicode, whitespace, code blocks, special chars) | 40 | 40 | 0 | PASS |
| Stress Tests (huge files, tiny files, repeated content) | 14 | 14 | 0 | PASS |
| Real-World Memory Files (65 production files, read-only) | 65 | 65 | 0 | PASS |
| CLAUDE.md Files (12 project configs, read-only) | 12 | 12 | 0 | PASS |
| v1 Backward Compat (v1 codebook with v2 engine) | 12 | 12 | 0 | PASS |
| Production Codebook Compat | 1 | 1 | 0 | PASS |
| **TOTAL** | **144** | **144** | **0** | **ALL PASS** |

## v1 vs v2 Comparison (Benchmark Corpus)

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| Token Savings | -1.7% | +2.1% | **+3.8 pts** |
| Traversal Speed | 7.34x | 12.03x | **+64%** |
| Round-Trip Accuracy | 100% | 100% | same |
| Codebook Entries | 90 | 76 | -15% smaller |
| Negative-savings entries | unknown | 0 | all entries save tokens |

## Real-World Token Savings (65 Production Memory Files)

**Total: 2.4% token savings (70,014 → 68,317 = 1,697 tokens saved)**
**Total: 2.5% byte savings (249,111 → 242,953 = 6,158 bytes saved)**

### Top 10 Files by Token Savings

| File | Tokens | Comp Tokens | Savings |
|------|--------|-------------|---------|
| file_01 (urls/config) | 197 | 163 | **17.3%** |
| file_02 (urls/links) | 1,666 | 1,415 | **15.1%** |
| file_03 (api endpoints) | 756 | 684 | **9.5%** |
| file_04 (tool config) | 623 | 565 | **9.3%** |
| file_05 (promo tracking) | 471 | 433 | **8.1%** |
| file_06 (project notes) | 760 | 704 | **7.4%** |
| file_07 (feedback rule) | 143 | 133 | **7.0%** |
| file_08 (tool notes) | 511 | 479 | **6.3%** |
| file_09 (session state) | 568 | 532 | **6.3%** |
| file_10 (project index) | 2,002 | 1,921 | **4.0%** |

### Files with Negative/Zero Savings (3 of 65)

| File | Tokens | Comp Tokens | Savings | Why |
|------|--------|-------------|---------|-----|
| file_37 (profile) | 379 | 380 | -0.3% | Small file, mostly prose, no paths |
| file_59 (session checkpoint) | 391 | 392 | -0.3% | Session-specific, few repeating patterns |
| file_53 (project plan) | 1,399 | 1,413 | -1.0% | Dense technical content, entity codes cost slightly more |

### Pattern Analysis

Files with **high path/URL density** see 5-17% token savings.
Files with **mostly prose** see 0-3% savings.
Files with **very few codebook matches** see ~0% or slight negative.

The v2 codebook was built from the same files being compressed. In production, the codebook would be trained once and reused, meaning the per-file overhead ratio would improve with more files.

## Edge Case Tests (40 tests)

All passing. Categories tested:
- Empty/whitespace: empty string, spaces, newlines, tabs (8 tests)
- Unicode: emoji, CJK, accented, math symbols, zero-width, private use area (6 tests)
- Code blocks: fenced, inline with $, regex, nested backticks (4 tests)
- Markdown: headers, bold/italic, links, tables, blockquotes, task lists, HR (7 tests)
- Code-like strings: literal $K, $P1, $RR, dollar amounts, hash tags, fake codes (7 tests)
- Special characters: backslashes, angle brackets, JSON, pipes, regex metacharacters, all ASCII (6 tests)
- Headers: text with fake PackRat v1/v2 headers already present (2 tests)

## Stress Tests (14 tests)

All passing:
- Repeated word 200x, repeated path 100x, repeated phrase 50x
- 10K char single line, 50K char file
- Very long path (50 nested subdirs), very long URL
- Tiny 1-char and 2-char files
- Only numbers, only punctuation
- Alternating entities + paths (50 lines)
- Null bytes, high-byte characters

## Security: Secrets Filter

The v2 `learn()` function strips lines containing credential patterns before extraction:
- API keys, bot tokens, private keys, passwords
- Telegram bot tokens (format: `digits:alphanumeric`)
- Stripe keys (`sk_live_*`, `sk_test_*`)
- NVIDIA keys (`nvapi-*`)
- GitHub tokens (`ghp_*`)
- Long hex strings (wallet addresses, hashes)
- RevenueCat keys (`appl_*`), Supadata keys (`sd_*`)

Verified: No secrets leaked into any generated codebook during testing.

## Backward Compatibility

- v2 engine correctly loads and uses v1 codebooks (version 1 format)
- v2 engine correctly loads and uses a real production codebook
- v1 codebook data structure is auto-upgraded (adds `paths: {}` section)
- v2 header (`<!-- packrat:v2 -->`) doesn't break v1 decompression pattern

## What Changed in v2

1. **Path/URL extraction** — Regex-based capture of file paths and URLs during `learn()`. Stored in new `paths` codebook section. Each path saves 6-19 tokens per occurrence.
2. **Token-aware scoring** — Uses tiktoken (cl100k_base) to verify actual token savings before adding entries. Entries that cost tokens are rejected.
3. **Secrets filter** — Strips lines containing credential-like patterns before phrase/entity extraction.
4. **Phrase deduplication** — Removes phrases that are substrings of higher-scoring phrases.
5. **Entity cleanup** — Skips mangled URL fragments, hex hashes, and extremely long CamelCase strings.
6. **Batch token counting** — New `--batch` JSON mode in count_tokens.py for ~10x faster learn().
7. **Auto-prune** — Existing codebook entries that cost tokens are removed during learn().
