/**
 * PackRat v2 — Token-Optimized Auto-Learning Codebook Compression
 * Makes AI agent memory files smaller AND cheaper (fewer tokens).
 *
 * v2 CHANGES:
 *   - Token-aware codebook scoring (uses tiktoken to verify actual savings)
 *   - Path/URL extraction (biggest token savings: 6-19 tokens per hit)
 *   - Boilerplate/section header capture
 *   - Hierarchical path compression (base prefix + relative paths)
 *   - Auto-prune entries that cost more tokens than they save
 *   - Negative-savings entries never added to codebook
 *
 * Zero deps. Pure JS. Works with Node 18+ and Bun.
 *
 * @example
 *   import { Codebook, compress, decompress, learn } from 'packrat-compress';
 *
 *   const cb = new Codebook('.packrat/codebook.json');
 *   learn(['.'], cb);                    // scan files, build codebook
 *   const out = compress(text, cb);      // compress
 *   const orig = decompress(out, cb);    // decompress
 */

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

// ─── Constants ───

const HEADER_RE = /^<!--\s*packrat:v(\d+)\s+codebook:(.+?)\s*-->/m;
const ESC_DOLLAR = '\uE000'; // Escape marker for $ codes
const ESC_HASH = '\uE001';   // Escape marker for # codes

const STOP_WORDS = new Set([
  'a','an','the','and','or','but','in','on','at','to','for','of','is','it',
  'was','are','be','been','do','does','did','will','would','could','should',
  'may','might','can','shall','have','has','had','with','from','by','not',
  'very','just','also','all','each','both','few','more','most','other','some',
  'no','only','so','than','too','while','what','how','where','when','who',
  'which','about','that','this','these','those','i','you','he','she','we',
  'they','me','him','her','us','them','my','your','his','its','our','their',
  'ok','ya','yeah','lol','like','gonna','wanna','dont','im','ive','youre',
  'thats','lets','hey','oh','well','sure','right','got','get','go','went',
  'thing','stuff','something','anything','here','there','then','now','still',
  'been','being','if','else','up','down','out','into','over','after','before',
]);

const COMMON_WORDS = new Set([
  'let','the','this','that','here','there','what','when','where','how','why',
  'now','found','created','built','edit','made','fixed','added','removed',
  'changed','updated','set','got','used','need','make','take','give','keep',
  'start','stop','run','check','look','see','try','done','works','good',
  'new','old','first','last','next','best','file','files','code','test',
]);

// v2: Secrets patterns — lines matching these are STRIPPED before phrase extraction
// Catches API keys, bot tokens, wallet hashes, and common credential patterns
// v2: Secrets patterns — lines matching these are STRIPPED before extraction
// Catches API keys, bot tokens, wallet hashes, credential patterns, and key-like strings
const SECRETS_RE = /(?:API[_ ]?KEY|BOT[_ ]?TOKEN|_TOKEN|_SECRET|PASSWORD|PRIVATE[_ ]?KEY|ACCESS[_ ]?TOKEN|AUTH[_ ]?TOKEN|BEARER|[Pp]exels key|[Ss]tripe|sk_live|sk_test|nvapi-|appl_[A-Za-z]{10,}|sd_[a-f0-9]{20,}|sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|[0-9a-f]{40,}|\d{10}:[A-Za-z0-9_-]{30,}|Production key|Supadata API)/i;

// Python path for tiktoken (Windows-friendly)
const PYTHON_PATH = process.env.PYTHON_PATH
  || process.env.PACKRAT_PYTHON
  || 'python';

// ─── Token Counting ───

// Cache for token counts to avoid repeated Python calls
const _tokenCache = new Map();

/**
 * Count tokens using tiktoken cl100k_base.
 * Falls back to chars/4 estimate if Python/tiktoken unavailable.
 * @param {string} text
 * @returns {number}
 */
export function countTokens(text) {
  if (_tokenCache.has(text)) return _tokenCache.get(text);

  let count;
  try {
    // Find the count_tokens.py helper
    const scriptPaths = [
      path.join(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/, '$1')), 'benchmark', 'count_tokens.py'),
      path.join(process.cwd(), 'benchmark', 'count_tokens.py'),
    ];
    let scriptPath;
    for (const p of scriptPaths) {
      if (fs.existsSync(p)) { scriptPath = p; break; }
    }

    if (scriptPath) {
      // execFileSync — no shell, so PYTHON_PATH can't be command-injected
      // even if the env var contains quotes, spaces, or shell metacharacters.
      const result = execFileSync(
        PYTHON_PATH,
        [scriptPath],
        { input: text, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024, timeout: 10000 }
      );
      count = parseInt(result.trim());
    } else {
      count = Math.ceil(text.length / 3.5); // slightly better estimate
    }
  } catch {
    count = Math.ceil(text.length / 3.5);
  }

  _tokenCache.set(text, count);
  return count;
}

/**
 * Batch count tokens for multiple strings (more efficient).
 * @param {string[]} texts
 * @returns {number[]}
 */
function countTokensBatch(texts) {
  // Check cache first
  const uncached = [];
  const uncachedIdx = [];
  const results = new Array(texts.length);

  for (let i = 0; i < texts.length; i++) {
    if (_tokenCache.has(texts[i])) {
      results[i] = _tokenCache.get(texts[i]);
    } else {
      uncached.push(texts[i]);
      uncachedIdx.push(i);
    }
  }

  if (uncached.length === 0) return results;

  try {
    const scriptPaths = [
      path.join(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/, '$1')), 'benchmark', 'count_tokens.py'),
      path.join(process.cwd(), 'benchmark', 'count_tokens.py'),
    ];
    let scriptPath;
    for (const p of scriptPaths) {
      if (fs.existsSync(p)) { scriptPath = p; break; }
    }

    if (scriptPath) {
      // Use JSON batch mode: send array, get array back
      const batchInput = JSON.stringify(uncached);
      const result = execFileSync(
        PYTHON_PATH,
        [scriptPath, '--batch'],
        { input: batchInput, encoding: 'utf-8', maxBuffer: 50 * 1024 * 1024, timeout: 30000 }
      );
      const counts = JSON.parse(result.trim());
      for (let i = 0; i < uncached.length; i++) {
        _tokenCache.set(uncached[i], counts[i]);
        results[uncachedIdx[i]] = counts[i];
      }
    } else {
      for (let i = 0; i < uncached.length; i++) {
        const c = Math.ceil(uncached[i].length / 3.5);
        _tokenCache.set(uncached[i], c);
        results[uncachedIdx[i]] = c;
      }
    }
  } catch {
    for (let i = 0; i < uncached.length; i++) {
      const c = Math.ceil(uncached[i].length / 3.5);
      _tokenCache.set(uncached[i], c);
      results[uncachedIdx[i]] = c;
    }
  }

  return results;
}

// ─── Codebook ───

export class Codebook {
  constructor(codebookPath = '.packrat/codebook.json') {
    this.path = codebookPath;
    this.data = { version: 2, generated: '', entries: {}, phrases: {}, paths: {} };
  }

  load() {
    if (!fs.existsSync(this.path)) return false;
    try {
      this.data = JSON.parse(fs.readFileSync(this.path, 'utf-8'));
      // Ensure v2 sections exist
      if (!this.data.paths) this.data.paths = {};
      if (this.data.version < 2) this.data.version = 2;
      return true;
    } catch { return false; }
  }

  save() {
    const dir = path.dirname(this.path);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    this.data.generated = new Date().toISOString().slice(0, 10);
    fs.writeFileSync(this.path, JSON.stringify(this.data, null, 2));
  }

  allPairs() {
    const pairs = [];
    // Paths first (longest patterns, biggest savings)
    for (const [code, text] of Object.entries(this.data.paths)) pairs.push({ code, text, type: 'path' });
    for (const [code, text] of Object.entries(this.data.entries)) pairs.push({ code, text, type: 'entity' });
    for (const [code, text] of Object.entries(this.data.phrases)) pairs.push({ code, text, type: 'phrase' });
    pairs.sort((a, b) => b.text.length - a.text.length);
    return pairs;
  }

  get size() {
    return Object.keys(this.data.entries).length
      + Object.keys(this.data.phrases).length
      + Object.keys(this.data.paths).length;
  }

  hasCode(code) {
    return code in this.data.entries || code in this.data.phrases || code in this.data.paths;
  }

  hasText(text) {
    const lower = text.toLowerCase();
    for (const v of Object.values(this.data.entries)) { if (v.toLowerCase() === lower) return true; }
    for (const v of Object.values(this.data.phrases)) { if (v.toLowerCase() === lower) return true; }
    for (const v of Object.values(this.data.paths)) { if (v === text) return true; } // paths are case-sensitive
    return false;
  }
}

// ─── Learn ───

/**
 * Scan files and update codebook with new patterns.
 * v2: token-aware scoring, path/URL extraction, boilerplate capture.
 *
 * @param {string[]} dirs - Directories to scan
 * @param {Codebook} codebook
 * @param {object} opts
 * @returns {{ newEntries: number, filesScanned: number, pruned: number, tokenStats: object }}
 */
export function learn(dirs, codebook, opts = {}) {
  const {
    maxEntities = 40,
    maxPhrases = 30,
    maxPaths = 20,
    minFreq = 2,
    minTokenSavings = 1, // minimum token savings per replacement to keep entry
    extensions = ['.md', '.txt'],
    tokenAware = true,   // v2: enable token-aware scoring
  } = opts;
  const extSet = new Set(extensions);

  // Walk directories and collect text
  const texts = [];
  function walk(dir) {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith('.') && entry.name !== '.claude') continue;
      if (entry.name === 'node_modules') continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (extSet.has(path.extname(entry.name).toLowerCase())) {
        try { texts.push(fs.readFileSync(full, 'utf-8')); } catch {}
      }
    }
  }
  for (const d of dirs) walk(d);

  // v2: Strip lines containing secrets/credentials before any extraction
  const allLines = texts.join('\n').split('\n');
  const safeLines = allLines.filter(line => !SECRETS_RE.test(line));
  const allText = safeLines.join('\n');
  const sizeBefore = codebook.size;

  // ══════════════════════════════════════════════
  // ── v2: PATH EXTRACTION (biggest token wins) ──
  // ══════════════════════════════════════════════

  const pathFreq = {};

  // Match file paths (Windows and Unix)
  const pathMatches = allText.match(/(?:[A-Z]:[\\/]|~\/|\.\/|\/(?:home|usr|var|opt|etc)\/)[^\s"'`<>)}\]]+/g) || [];
  for (const p of pathMatches) {
    const clean = p.replace(/[`'")\]},;]+$/, ''); // strip trailing punctuation
    if (clean.length < 10) continue;
    pathFreq[clean] = (pathFreq[clean] || 0) + 1;
  }

  // Match URLs
  const urlMatches = allText.match(/https?:\/\/[^\s"'`<>)}\]]+/g) || [];
  for (const u of urlMatches) {
    const clean = u.replace(/[`'")\]},;]+$/, '');
    if (clean.length < 15) continue;
    pathFreq[clean] = (pathFreq[clean] || 0) + 1;
  }

  // Also find common path PREFIXES (e.g., C:/Users/dev/ appears in many different full paths)
  const prefixFreq = {};
  for (const p of Object.keys(pathFreq)) {
    // Split path and try progressively longer prefixes
    const parts = p.replace(/\\/g, '/').split('/');
    for (let len = 2; len < parts.length; len++) {
      const prefix = parts.slice(0, len).join('/') + '/';
      if (prefix.length < 8) continue;
      // Count how many full paths start with this prefix
      let count = 0;
      for (const [fp, freq] of Object.entries(pathFreq)) {
        if (fp.replace(/\\/g, '/').startsWith(prefix)) count += freq;
      }
      if (count >= 2) {
        prefixFreq[prefix] = count;
      }
    }
  }

  // Combine: individual paths + high-value prefixes
  const pathCandidates = [];
  for (const [p, freq] of Object.entries(pathFreq)) {
    if (freq >= minFreq) pathCandidates.push({ text: p, freq });
  }
  for (const [p, freq] of Object.entries(prefixFreq)) {
    if (freq >= 3 && !pathFreq[p]) pathCandidates.push({ text: p, freq });
  }

  // Score paths by token savings * frequency
  if (tokenAware && pathCandidates.length > 0) {
    const pathTexts = pathCandidates.map(c => c.text);
    const pathTokenCounts = countTokensBatch(pathTexts);
    for (let i = 0; i < pathCandidates.length; i++) {
      const codeTokens = 3; // most $XYZ codes cost ~3 tokens
      pathCandidates[i].tokenSavings = pathTokenCounts[i] - codeTokens;
      pathCandidates[i].score = pathCandidates[i].tokenSavings * pathCandidates[i].freq;
    }
    pathCandidates.sort((a, b) => b.score - a.score);
  } else {
    pathCandidates.sort((a, b) => (b.freq * b.text.length) - (a.freq * a.text.length));
  }

  // Add top paths to codebook
  let pidxPath = Object.keys(codebook.data.paths).length;
  for (const { text, tokenSavings } of pathCandidates.slice(0, maxPaths)) {
    if (codebook.hasText(text)) continue;
    if (tokenAware && tokenSavings !== undefined && tokenSavings < minTokenSavings) continue;
    const code = `$P${++pidxPath}`;
    if (!codebook.hasCode(code)) codebook.data.paths[code] = text;
  }

  // ══════════════════════════════════════════════
  // ── ENTITY EXTRACTION (token-aware v2) ────────
  // ══════════════════════════════════════════════

  const entityFreq = {};
  for (const raw of allText.split(/\s+/)) {
    const clean = raw.replace(/[^a-zA-Z0-9._-]/g, '');
    if (clean.length < 4) continue;
    const lower = clean.toLowerCase();
    if (STOP_WORDS.has(lower) || COMMON_WORDS.has(lower)) continue;
    if (/^\d/.test(clean) || /[.!?,;:)]$/.test(clean)) continue;
    // v2: Skip mangled path/URL fragments (contain multiple dots or look like joined paths)
    if ((clean.match(/\./g) || []).length >= 3) continue;
    if (clean.length > 30 && /[a-z][A-Z]/.test(clean)) continue; // likely CamelCase gibberish from URL
    // Skip hex hashes and long alphanumeric strings (likely secrets/IDs)
    if (/^[a-f0-9]{16,}$/i.test(clean)) continue;

    const isCap = clean[0] === clean[0].toUpperCase() && clean[0] !== clean[0].toLowerCase();
    const isTech = /[._-]/.test(clean) || (/[a-z]/.test(clean) && /[A-Z]/.test(clean.slice(1)));
    if (isCap || isTech) entityFreq[clean] = (entityFreq[clean] || 0) + 1;
  }

  // Merge case-insensitive
  const merged = {};
  for (const [w, f] of Object.entries(entityFreq)) {
    const lower = w.toLowerCase();
    if (!merged[lower] || f > merged[lower].f) merged[lower] = { w, f };
    else merged[lower].f += f;
  }

  let entities = Object.values(merged).filter(e => e.f >= minFreq);

  // v2: Token-aware scoring
  if (tokenAware && entities.length > 0) {
    const entityTexts = entities.map(e => e.w);
    const entityCodes = entities.map(e => '$' + _genCode(e.w).slice(1));
    const textTokens = countTokensBatch(entityTexts);
    const codeTokens = countTokensBatch(entityCodes);

    for (let i = 0; i < entities.length; i++) {
      entities[i].tokenSavings = textTokens[i] - codeTokens[i];
      entities[i].score = entities[i].tokenSavings * entities[i].f;
    }

    // Filter out entries that cost tokens (negative savings)
    entities = entities.filter(e => e.tokenSavings >= minTokenSavings);
    entities.sort((a, b) => b.score - a.score);
  } else {
    entities.sort((a, b) => (b.f * b.w.length) - (a.f * a.w.length));
  }

  for (const { w } of entities.slice(0, maxEntities)) {
    if (codebook.hasText(w)) continue;
    let code = _genCode(w);
    let attempt = 0;
    while (codebook.hasCode(code) && attempt < 5) {
      code += (w[Math.min(attempt + 3, w.length - 1)] || String(attempt)).toUpperCase();
      attempt++;
    }
    if (!codebook.hasCode(code)) codebook.data.entries[code] = w;
  }

  // ══════════════════════════════════════════════
  // ── PHRASE EXTRACTION (v2: includes boilerplate) ──
  // ══════════════════════════════════════════════

  const phraseFreq = {};

  // Standard phrase extraction
  for (const sentence of allText.split(/[.!?\n]+/)) {
    const tokens = sentence.trim().split(/\s+/).filter(w => w.length > 0);
    for (let n = 2; n <= 5; n++) {  // v2: up to 5-grams (was 4)
      for (let i = 0; i <= tokens.length - n; i++) {
        const phrase = tokens.slice(i, i + n).join(' ');
        const meaningful = tokens.slice(i, i + n).filter(w => {
          const l = w.toLowerCase().replace(/[^a-z]/g, '');
          return l.length > 1 && !STOP_WORDS.has(l);
        });
        if (meaningful.length < 1 || phrase.length < 8) continue;
        phraseFreq[phrase] = (phraseFreq[phrase] || 0) + 1;
      }
    }
  }

  // v2: Capture markdown boilerplate patterns
  const boilerplatePatterns = allText.match(/^#{1,3}\s+[A-Z][^\n]{5,60}$/gm) || [];
  for (const bp of boilerplatePatterns) {
    const trimmed = bp.trim();
    if (trimmed.length >= 8) phraseFreq[trimmed] = (phraseFreq[trimmed] || 0) + 1;
  }
  // Bold patterns like **Status:**, **Path:**, **Note:**
  const boldPatterns = allText.match(/\*\*[A-Z][a-zA-Z ]{2,30}:\*\*/g) || [];
  for (const bp of boldPatterns) {
    phraseFreq[bp] = (phraseFreq[bp] || 0) + 1;
  }

  // Filter phrases
  const PHRASE_BLACKLIST = /^(let me|i'll|user:|assistant:|agent:|- user|- agent|# conversation|conversation summary|memory from|saved to vault|you're right|from earlier)/i;

  let phraseCandidates = Object.entries(phraseFreq)
    .filter(([p, f]) => f >= minFreq && !PHRASE_BLACKLIST.test(p));

  // v2: Token-aware phrase scoring
  if (tokenAware && phraseCandidates.length > 0) {
    const phraseTexts = phraseCandidates.map(([p]) => p);
    const phraseCodes = phraseCandidates.map((_, i) => `$p${i + 1}`);
    const textTokens = countTokensBatch(phraseTexts);
    const codeTokens = countTokensBatch(phraseCodes);

    phraseCandidates = phraseCandidates
      .map(([p, f], i) => ({
        phrase: p,
        freq: f,
        tokenSavings: textTokens[i] - codeTokens[i],
        score: (textTokens[i] - codeTokens[i]) * f,
      }))
      .filter(c => c.tokenSavings >= minTokenSavings)
      .sort((a, b) => b.score - a.score)
      .slice(0, maxPhrases);
  } else {
    phraseCandidates = phraseCandidates
      .sort((a, b) => (b[1] * b[0].length) - (a[1] * a[0].length))
      .slice(0, maxPhrases)
      .map(([p, f]) => ({ phrase: p, freq: f }));
  }

  // v2: Deduplicate overlapping phrases — remove substrings of higher-scored phrases
  const dedupedPhrases = [];
  for (const candidate of phraseCandidates) {
    const isSubstring = dedupedPhrases.some(kept => kept.phrase.includes(candidate.phrase));
    if (!isSubstring) dedupedPhrases.push(candidate);
  }

  let pidx = Object.keys(codebook.data.phrases).length;
  for (const { phrase } of dedupedPhrases) {
    if (codebook.hasText(phrase)) continue;
    codebook.data.phrases[`$p${++pidx}`] = phrase;
  }

  // ══════════════════════════════════════════════
  // ── v2: PRUNE NEGATIVE-SAVINGS ENTRIES ────────
  // ══════════════════════════════════════════════

  let pruned = 0;
  if (tokenAware) {
    const allEntries = [
      ...Object.entries(codebook.data.entries).map(([c, t]) => ({ section: 'entries', code: c, text: t })),
      ...Object.entries(codebook.data.phrases).map(([c, t]) => ({ section: 'phrases', code: c, text: t })),
      ...Object.entries(codebook.data.paths).map(([c, t]) => ({ section: 'paths', code: c, text: t })),
    ];

    if (allEntries.length > 0) {
      const entryTexts = allEntries.map(e => e.text);
      const entryCodes = allEntries.map(e => e.code);
      const textToks = countTokensBatch(entryTexts);
      const codeToks = countTokensBatch(entryCodes);

      for (let i = 0; i < allEntries.length; i++) {
        const savings = textToks[i] - codeToks[i];
        if (savings < minTokenSavings) {
          delete codebook.data[allEntries[i].section][allEntries[i].code];
          pruned++;
        }
      }
    }
  }

  codebook.save();

  const tokenStats = tokenAware ? {
    pathsAdded: Object.keys(codebook.data.paths).length,
    entitiesKept: Object.keys(codebook.data.entries).length,
    phrasesKept: Object.keys(codebook.data.phrases).length,
    pruned,
  } : null;

  return { newEntries: codebook.size - sizeBefore + pruned, filesScanned: texts.length, pruned, tokenStats };
}

// ─── Compress ───

/**
 * Compress text using codebook.
 * @param {string} text - Raw text
 * @param {Codebook} codebook
 * @returns {string} Compressed text with PackRat header
 */
export function compress(text, codebook) {
  const pairs = codebook.allPairs();
  let out = text;

  // Escape literal code-like tokens BEFORE compression
  const allCodes = pairs.map(p => p.code);
  const sortedCodes = [...allCodes].sort((a, b) => b.length - a.length);
  for (const code of sortedCodes) {
    const esc = code.startsWith('#') ? ESC_HASH : ESC_DOLLAR;
    const escaped = _escRe(code);
    out = out.replace(new RegExp(escaped, 'g'), `${esc}${code.slice(1)}`);
  }

  for (const { code, text: pattern, type } of pairs) {
    const escaped = _escRe(pattern);
    // Paths and phrases: no word boundary (they contain special chars)
    // Entities: use word boundary for clean matching
    const re = (type === 'path' || type === 'phrase' || code.startsWith('$p'))
      ? new RegExp(escaped, 'g')
      : new RegExp(`\\b${escaped}\\b`, 'g');
    out = out.replace(re, code);
  }
  return `<!-- packrat:v2 codebook:${codebook.path} -->\n${out}`;
}

/**
 * Compress a file in-place or to .pr.md.
 */
export function compressFile(filePath, codebook, inPlace = false) {
  const original = fs.readFileSync(filePath, 'utf-8');
  const compressed = compress(original, codebook);
  const outPath = inPlace ? filePath : filePath.replace(/\.md$/, '.pr.md');
  fs.writeFileSync(outPath, compressed);
  const ratio = (original.length / (compressed.length - compressed.indexOf('\n'))).toFixed(1);
  return { outPath, originalSize: original.length, compressedSize: compressed.length, ratio: ratio + 'x' };
}

// ─── Decompress ───

/**
 * Decompress PackRat-compressed text.
 */
export function decompress(compressed, codebook) {
  // Strip header line only (preserve all other whitespace)
  let text = compressed.replace(/^<!--\s*packrat:v\d+\s+codebook:.+?-->\n?/, '');
  // Sort by LONGEST CODE FIRST to prevent prefix collisions
  const pairs = codebook.allPairs().sort((a, b) => b.code.length - a.code.length);
  for (const { code, text: pattern } of pairs) {
    text = text.replace(new RegExp(_escRe(code), 'g'), pattern);
  }
  // Unescape literal code tokens
  const allCodes = codebook.allPairs().map(p => p.code);
  const sortedCodes = [...allCodes].sort((a, b) => b.length - a.length);
  for (const code of sortedCodes) {
    const esc = code.startsWith('#') ? ESC_HASH : ESC_DOLLAR;
    const escapedForm = `${esc}${code.slice(1)}`;
    text = text.replace(new RegExp(_escRe(escapedForm), 'g'), code);
  }
  return text;
}

// ─── Stats ───

/**
 * Get codebook statistics (v2: includes token analysis).
 */
export function stats(codebook) {
  const sections = {
    paths: Object.entries(codebook.data.paths || {}),
    entities: Object.entries(codebook.data.entries),
    phrases: Object.entries(codebook.data.phrases),
  };

  const all = [...sections.paths, ...sections.entities, ...sections.phrases];
  all.sort((a, b) => (b[1].length - b[0].length) - (a[1].length - a[0].length));

  // Token analysis for each entry
  const tokenDetails = [];
  if (all.length > 0) {
    const texts = all.map(([, t]) => t);
    const codes = all.map(([c]) => c);
    const textToks = countTokensBatch(texts);
    const codeToks = countTokensBatch(codes);
    for (let i = 0; i < all.length; i++) {
      tokenDetails.push({
        code: all[i][0],
        text: all[i][1],
        charSavings: all[i][1].length - all[i][0].length,
        tokenSavings: textToks[i] - codeToks[i],
        textTokens: textToks[i],
        codeTokens: codeToks[i],
      });
    }
    tokenDetails.sort((a, b) => b.tokenSavings - a.tokenSavings);
  }

  return {
    version: codebook.data.version,
    generated: codebook.data.generated,
    paths: sections.paths.length,
    entities: Object.keys(codebook.data.entries).length,
    phrases: Object.keys(codebook.data.phrases).length,
    total: codebook.size,
    topByTokenSavings: tokenDetails.slice(0, 15),
    negativeEntries: tokenDetails.filter(d => d.tokenSavings < 0),
  };
}

// ─── Helpers ───

function _escRe(str) { return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

function _genCode(word) {
  const clean = word.replace(/[^a-zA-Z]/g, '');
  if (clean.length <= 3) return '$' + clean.toUpperCase();
  const first = clean[0].toUpperCase();
  const cons = clean.slice(1).replace(/[aeiou]/gi, '');
  return '$' + first + (cons.length >= 2 ? cons.slice(0, 2).toUpperCase() : clean.slice(1, 3).toUpperCase());
}
