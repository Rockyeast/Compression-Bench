# PackRat

**Auto-learning codebook compression for AI agent memory. Token-optimized.**

Makes context files smaller, cheaper, and 12x faster to search — while staying fully readable by any LLM. No decoder needed. 100% lossless.

## Why PackRat

AI agents forget because memory bloats their context window. Every token costs money and latency. PackRat compresses memory files with a self-learning codebook that any LLM reads natively.

**v2 is token-aware** — it uses tiktoken to verify that every codebook entry actually saves tokens, not just characters. Entries that cost more tokens than they save are automatically rejected.

### Benchmark Results (real-world, 65 production files)

| Metric | Result |
|--------|--------|
| Token savings | **2.4% average, up to 17% on path-heavy files** |
| Byte savings | **2.5% average** |
| Search speed | **12x faster** across compressed files |
| Round-trip accuracy | **100%** (144 tests, 0 failures) |
| Codebook overhead | 76 entries, zero negative-savings |

> MemPalace (23K stars) drops to 84.2% accuracy with compression.
> PackRat: **100% accuracy. Always. Lossless is non-negotiable.**

**[Full benchmark data with per-file results](benchmark/BENCHMARKS.md)**

## How It Works

1. **Learn** — Scans your files. Extracts repeated paths, URLs, entities, and phrases. Scores each by `frequency x token_savings` using tiktoken.
2. **Compress** — Replaces patterns with short codes. Every file gets a self-documenting header.
3. **Read** — Any LLM reads the codebook header and understands the compressed file cold. No fine-tuning, no decoder.

```
Original:  "Updated recipe import at C:/Users/dev/projects/myapp/scripts/import.mjs"
Compressed: "Updated recipe import at $P3/scripts/import.mjs"

Original:  "Deploy to Cloudflare Pages using TypeScript and React with Supabase"
Compressed: "Deploy to $CF Pages using $TS and $RCT with $SB"
```

The codebook: `$P3 = C:/Users/dev/projects/myapp`, `$CF = Cloudflare`, etc.

### Where the Savings Come From

| Pattern Type | Example | Tokens Saved Per Hit |
|-------------|---------|---------------------|
| File paths | `C:/Users/dev/projects/app/src/` | **6-19 tokens** |
| URLs | `https://github.com/user/repo` | **8-14 tokens** |
| Long tech names | `ImageGen_portable` | **5-8 tokens** |
| Repeated phrases | `via OpenRouter for free` | **2-3 tokens** |

PackRat v2 automatically skips words that tokenizers already handle efficiently (like "JavaScript" = 1 token). Only patterns that **actually save tokens** make it into the codebook.

## Perfect for Obsidian + AI Workflows

If you use Obsidian as a second brain and feed your vault to AI agents, PackRat is built for you:

- **Learn** from your entire vault — hundreds of markdown files
- **Compress** before loading into agent context windows
- **Save tokens** on every single API call
- **Search 12x faster** across your compressed knowledge base
- **Zero data loss** — decompress back to the exact original anytime

## Install

```bash
npm install -g packrat-compress
```

Or use directly:

```bash
npx packrat-compress init
```

## Usage

```bash
# Initialize in your project
packrat init

# Scan files and build codebook (token-aware by default)
packrat learn .
packrat learn ./memory/
packrat learn ./obsidian-vault/

# Compress a file
packrat compress context.md
# -> context.pr.md (compressed)

# Decompress for humans
packrat decompress context.pr.md
# -> context.expanded.md

# Check stats
packrat stats
```

## Programmatic API

```js
import { Codebook, learn, compress, decompress, stats, countTokens } from 'packrat-compress';

// Load or create codebook
const cb = new Codebook('.packrat/codebook.json');
cb.load();

// Learn from files (v2: token-aware by default)
learn(['.'], cb, { tokenAware: true });

// Compress text
const compressed = compress('Your long context text here...', cb);

// Decompress (100% lossless)
const original = decompress(compressed, cb);

// Stats (v2: includes token analysis per entry)
const info = stats(cb);
console.log(`${info.total} entries: ${info.paths} paths, ${info.entities} entities, ${info.phrases} phrases`);
```

### Learn Options

```js
learn(dirs, codebook, {
  tokenAware: true,      // Use tiktoken to verify actual token savings (default: true)
  minTokenSavings: 1,    // Minimum tokens saved per replacement to keep entry (default: 1)
  maxPaths: 20,          // Max path/URL entries (default: 20)
  maxEntities: 40,       // Max entity entries (default: 40)
  maxPhrases: 30,        // Max phrase entries (default: 30)
  minFreq: 2,            // Minimum frequency to consider (default: 2)
  extensions: ['.md', '.txt'],  // File extensions to scan
});
```

## Compressed File Format

Every PackRat-compressed file starts with a header:

```markdown
<!-- packrat:v2 codebook:.packrat/codebook.json -->
```

This tells any LLM: "load that codebook, then read this file." The format is self-documenting.

## Codebook Format (v2)

```json
{
  "version": 2,
  "generated": "2026-04-10",
  "paths": {
    "$P1": "C:/Users/dev/projects/myapp/",
    "$P2": "https://github.com/user/repo"
  },
  "entries": {
    "$K": "Kevin",
    "$CF": "Cloudflare",
    "$TS": "TypeScript"
  },
  "phrases": {
    "$p1": "via OpenRouter for free",
    "$p2": "## Critical Reminders"
  }
}
```

- **Paths**: File paths and URLs (biggest token savings, 6-19 tokens each)
- **Entries**: Entities — people, tools, services, tech terms
- **Phrases**: Repeated multi-word patterns and markdown boilerplate
- Human-editable JSON — add your own entries anytime
- Fully backward-compatible with v1 codebooks

## Security

PackRat v2 includes a **secrets filter** that automatically strips lines containing credential patterns before learning:

- API keys, bot tokens, private keys, passwords
- Stripe keys, GitHub tokens, NVIDIA keys
- Long hex strings (wallet addresses, hashes)
- Common credential formats (`sk_live_*`, `ghp_*`, `nvapi-*`)

Your codebook will never contain leaked secrets, even if your source files do.

## Design Principles

- **Zero dependencies** — Pure Node.js, works with Node 18+
- **100% lossless** — Perfect round-trip on every file, every time
- **Token-optimized** — Every codebook entry verified to save actual tokens
- **Self-documenting** — Codebook reference in every compressed file
- **LLM-native** — Any model reads compressed files without fine-tuning
- **Secure** — Built-in secrets filter prevents credential leaks
- **Backward-compatible** — v2 engine reads v1 codebooks seamlessly

## Claude Code Skill

PackRat is also available as a Claude Code skill:

```
/packrat init
/packrat learn ./memory
/packrat compress context.md
/packrat stats
```

## License

MIT — Kevin Cline / [ClawdWorks](https://github.com/kevdogg102396-afk)
