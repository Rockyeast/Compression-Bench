#!/usr/bin/env node
/**
 * Mega File Benchmark — tests PackRat on a single large memory file
 */
import fs from 'node:fs';
import { execSync } from 'node:child_process';
import { Codebook, learn, compress, decompress } from '../index.mjs';
import { performance } from 'node:perf_hooks';

const PYTHON = process.env.PYTHON_PATH || 'python';

function countTokens(text) {
  const result = execSync(`"${PYTHON}" benchmark/count_tokens.py`, {
    input: text, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024
  });
  return parseInt(result.trim());
}

// Train fresh codebook on the mega file
const cb = new Codebook('benchmark/corpus/.packrat/codebook-mega.json');
const result = learn(['benchmark/corpus'], cb, {
  maxEntities: 60,
  maxPhrases: 0,
  minFreq: 5,
  extensions: ['.md']
});

// Prune low-value entries
let pruned = 0;
for (const [code, text] of Object.entries(cb.data.entries)) {
  if (text.length - code.length < 3 || text.includes('http')) {
    delete cb.data.entries[code];
    pruned++;
  }
}
cb.save();

console.log(`Codebook: ${cb.size} entries (pruned ${pruned})`);
console.log('');

// Load mega file
const mega = fs.readFileSync('benchmark/corpus/MEGA_MEMORY.md', 'utf-8');

// Compress
const t0 = performance.now();
const compressed = compress(mega, cb);
const compressTime = performance.now() - t0;

// Decompress
const t1 = performance.now();
const decompressed = decompress(compressed, cb);
const decompressTime = performance.now() - t1;

// Round-trip check
const roundTrip = decompressed === mega;

// Strip header for measurement
const compBody = compressed.replace(/^<!--.*?-->\n/, '');

// Token counts
console.log('Counting tokens (this takes a moment on 251KB)...');
const origTokens = countTokens(mega);
const compTokens = countTokens(compBody);
const cbText = fs.readFileSync(cb.path, 'utf-8');
const cbTokens = countTokens(cbText);

// Search speed test
const queries = ['TypeScript', 'React', 'Supabase', 'Vercel', 'GitHub', 'Docker',
  'Telegram', 'Cloudflare', 'deploy', 'error', 'memory', 'API', 'config', 'build'];

let rawSearchTime = 0;
let compSearchTime = 0;
for (let iter = 0; iter < 100; iter++) {
  let t = performance.now();
  for (const q of queries) mega.match(new RegExp(q, 'gi'));
  rawSearchTime += performance.now() - t;

  t = performance.now();
  for (const q of queries) compBody.match(new RegExp(q, 'gi'));
  compSearchTime += performance.now() - t;
}
rawSearchTime /= 100;
compSearchTime /= 100;

console.log('');
console.log('╔══════════════════════════════════════════════════════╗');
console.log('║         PACKRAT MEGA FILE BENCHMARK                  ║');
console.log('║         251KB — All 65 Memory Files Combined         ║');
console.log('╚══════════════════════════════════════════════════════╝');
console.log('');
console.log(`  Round-Trip Accuracy:  ${roundTrip ? '✅ 100% PERFECT' : '❌ FAIL'}`);
console.log('');
console.log('  ── Size ──');
console.log(`  Original:    ${mega.length.toLocaleString()} bytes`);
console.log(`  Compressed:  ${compBody.length.toLocaleString()} bytes`);
console.log(`  Byte savings: ${((1 - compBody.length / mega.length) * 100).toFixed(1)}%`);
console.log(`  Ratio:       ${(mega.length / compBody.length).toFixed(2)}x`);
console.log('');
console.log('  ── Tokens (tiktoken cl100k_base) ──');
console.log(`  Original:    ${origTokens.toLocaleString()} tokens`);
console.log(`  Compressed:  ${compTokens.toLocaleString()} tokens`);
console.log(`  Codebook:    ${cbTokens.toLocaleString()} tokens`);
console.log(`  Token savings (without codebook): ${((1 - compTokens / origTokens) * 100).toFixed(1)}%`);
console.log(`  Token savings (with codebook):    ${((1 - (compTokens + cbTokens) / origTokens) * 100).toFixed(1)}%`);
console.log('');
console.log('  ── Speed ──');
console.log(`  Compress time:   ${compressTime.toFixed(1)}ms`);
console.log(`  Decompress time: ${decompressTime.toFixed(1)}ms`);
console.log(`  Search (raw):    ${rawSearchTime.toFixed(3)}ms avg`);
console.log(`  Search (comp):   ${compSearchTime.toFixed(3)}ms avg`);
console.log(`  Search speedup:  ${(rawSearchTime / compSearchTime).toFixed(2)}x`);
console.log('');
console.log('  ── Codebook ──');
console.log(`  Entries: ${cb.size}`);
console.log(`  Codebook size: ${cbText.length} bytes / ${cbTokens} tokens`);

// Show top entity hits
const pairs = cb.allPairs();
const hits = [];
for (const {code, text} of pairs) {
  const re = new RegExp(`\\b${text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'g');
  const m = mega.match(re) || [];
  hits.push({code, text, count: m.length, saved: m.length * (text.length - code.length)});
}
hits.sort((a,b) => b.saved - a.saved);
console.log('');
console.log('  ── Top Entity Hits ──');
for (const h of hits.slice(0, 15)) {
  console.log(`    ${h.code} → ${h.text}: ${h.count} hits, ${h.saved} chars saved`);
}
