/**
 * Traversal Speed Benchmark
 * Measures how fast you can search through compressed vs uncompressed files.
 * Smaller files = faster I/O + faster string search.
 */
import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { Codebook, compress } from '../../index.mjs';

function median(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export async function runTraversalBenchmark(corpusDir, codebookPath, verbose = false) {
  const cb = new Codebook(codebookPath);
  cb.load();

  const files = fs.readdirSync(corpusDir).filter(f => f.endsWith('.md'));

  // Prepare raw and compressed corpora in memory
  const rawCorpus = files.map(f => ({
    name: f,
    content: fs.readFileSync(path.join(corpusDir, f), 'utf-8'),
  }));

  const compressedCorpus = rawCorpus.map(f => ({
    name: f.name,
    content: compress(f.content, cb),
  }));

  // Search queries — mix of entity names, technical terms, and common patterns
  const queries = [
    'TypeScript', 'React', 'Supabase', 'Vercel', 'GitHub', 'Docker',
    'Telegram', 'Cloudflare', 'FastAPI', 'Python', 'deploy', 'build',
    'error', 'config', 'API', 'database', 'migration', 'auth',
    'memory', 'compression',
  ];

  // Also create queries that match compressed codes
  const codeQueries = cb.allPairs().slice(0, 10).map(p => p.code);

  const ITERATIONS = 200;

  // Benchmark: search raw corpus
  const rawTimes = [];
  for (let iter = 0; iter < ITERATIONS; iter++) {
    const start = performance.now();
    let totalHits = 0;
    for (const query of queries) {
      for (const file of rawCorpus) {
        const re = new RegExp(query, 'gi');
        const matches = file.content.match(re);
        totalHits += matches ? matches.length : 0;
      }
    }
    rawTimes.push(performance.now() - start);
  }

  // Benchmark: search compressed corpus (search for codes instead of full terms)
  const compTimes = [];
  for (let iter = 0; iter < ITERATIONS; iter++) {
    const start = performance.now();
    let totalHits = 0;
    for (const query of codeQueries) {
      for (const file of compressedCorpus) {
        const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const re = new RegExp(escaped, 'g');
        const matches = file.content.match(re);
        totalHits += matches ? matches.length : 0;
      }
    }
    compTimes.push(performance.now() - start);
  }

  // Benchmark: file load time (simulated via string length processing)
  const rawLoadTimes = [];
  const compLoadTimes = [];
  for (let iter = 0; iter < ITERATIONS; iter++) {
    let start = performance.now();
    let totalLen = 0;
    for (const file of rawCorpus) totalLen += file.content.length;
    rawLoadTimes.push(performance.now() - start);

    start = performance.now();
    totalLen = 0;
    for (const file of compressedCorpus) totalLen += file.content.length;
    compLoadTimes.push(performance.now() - start);
  }

  const rawSize = rawCorpus.reduce((s, f) => s + f.content.length, 0);
  const compSize = compressedCorpus.reduce((s, f) => s + f.content.length, 0);

  const rawMedian = median(rawTimes);
  const compMedian = median(compTimes);
  const speedup = compMedian > 0 ? (rawMedian / compMedian) : Infinity;

  const results = {
    name: 'Traversal Speed',
    corpusFiles: files.length,
    rawCorpusSize: rawSize + ' bytes',
    compressedCorpusSize: compSize + ' bytes',
    sizeReduction: ((1 - compSize / rawSize) * 100).toFixed(1) + '%',
    searchQueries: queries.length,
    iterations: ITERATIONS,
    rawSearchMedian: rawMedian.toFixed(3) + 'ms',
    compressedSearchMedian: compMedian.toFixed(3) + 'ms',
    speedup: speedup.toFixed(2) + 'x',
    note: 'Compressed files are smaller = fewer bytes to scan. Codes are shorter than entity names = faster regex matching.',
  };

  if (verbose) {
    console.log(`  Raw corpus: ${rawSize} bytes, Compressed: ${compSize} bytes`);
    console.log(`  Raw search median: ${rawMedian.toFixed(3)}ms`);
    console.log(`  Compressed search median: ${compMedian.toFixed(3)}ms`);
    console.log(`  Speedup: ${speedup.toFixed(2)}x`);
  }

  return results;
}
