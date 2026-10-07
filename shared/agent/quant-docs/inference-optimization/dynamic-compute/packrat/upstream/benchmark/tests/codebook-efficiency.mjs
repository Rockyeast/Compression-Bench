/**
 * Codebook Efficiency Test
 * Measures hit rates, dead entries, savings per code, and collisions.
 */
import fs from 'node:fs';
import path from 'node:path';
import { Codebook } from '../../index.mjs';

export async function runCodebookEfficiency(corpusDir, codebookPath, verbose = false) {
  const cb = new Codebook(codebookPath);
  cb.load();

  // Load all corpus text
  const files = fs.readdirSync(corpusDir).filter(f => f.endsWith('.md'));
  const allTexts = files.map(f => fs.readFileSync(path.join(corpusDir, f), 'utf-8'));
  const corpus = allTexts.join('\n');

  const pairs = cb.allPairs();
  const entryStats = [];

  for (const { code, text: pattern } of pairs) {
    // Count occurrences in corpus
    const escaped = pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const isPhrase = code.startsWith('$p');
    const re = isPhrase
      ? new RegExp(escaped, 'gi')
      : new RegExp(`\\b${escaped}\\b`, 'gi');

    const matches = corpus.match(re) || [];
    const hitCount = matches.length;
    const charsSaved = hitCount * (pattern.length - code.length);

    // Check for natural code collisions (code appearing in raw text as-is)
    const codeEscaped = code.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const collisionRe = new RegExp(codeEscaped, 'g');
    const collisions = corpus.match(collisionRe) || [];

    entryStats.push({
      code,
      text: pattern,
      type: isPhrase ? 'phrase' : 'entity',
      hits: hitCount,
      charsSaved,
      collisions: collisions.length,
    });
  }

  // Sort by savings (descending)
  const ranked = [...entryStats].sort((a, b) => b.charsSaved - a.charsSaved);

  // Categorize
  const active = entryStats.filter(e => e.hits > 0);
  const dead = entryStats.filter(e => e.hits === 0);
  const colliding = entryStats.filter(e => e.collisions > 0);
  const totalCharsSaved = entryStats.reduce((sum, e) => sum + Math.max(0, e.charsSaved), 0);
  const utilization = pairs.length > 0 ? (active.length / pairs.length * 100) : 0;

  if (verbose) {
    console.log('\n  Top 10 performers:');
    for (const e of ranked.slice(0, 10)) {
      console.log(`    ${e.code} → "${e.text}" | ${e.hits} hits, ${e.charsSaved} chars saved`);
    }
    if (dead.length > 0) {
      console.log(`\n  Dead entries (${dead.length}):`);
      for (const e of dead) {
        console.log(`    ${e.code} → "${e.text}" (0 hits)`);
      }
    }
    if (colliding.length > 0) {
      console.log(`\n  Potential collisions (${colliding.length}):`);
      for (const e of colliding) {
        console.log(`    ${e.code} appears ${e.collisions}x naturally in corpus`);
      }
    }
  }

  return {
    name: 'Codebook Efficiency',
    totalEntries: pairs.length,
    activeEntries: active.length,
    deadEntries: dead.length,
    utilization: utilization.toFixed(1) + '%',
    totalCharsSaved,
    collisions: colliding.length,
    topPerformers: ranked.slice(0, 10).map(e => ({
      code: e.code, text: e.text, hits: e.hits, saved: e.charsSaved,
    })),
    deadList: dead.map(e => ({ code: e.code, text: e.text })),
    collisionList: colliding.map(e => ({
      code: e.code, text: e.text, naturalOccurrences: e.collisions,
    })),
  };
}
