/**
 * Real Token Count Analysis
 * Uses tiktoken (cl100k_base) for accurate GPT-4/Claude token counts.
 */
import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { Codebook, compress } from '../../index.mjs';

const PYTHON = process.env.PYTHON_PATH || 'python';

function countTokens(text) {
  try {
    const result = execSync(
      `"${PYTHON}" benchmark/count_tokens.py`,
      { input: text, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024, cwd: path.join(path.dirname(new URL(import.meta.url).pathname.slice(1)), '../..') }
    );
    return parseInt(result.trim());
  } catch (e) {
    // Fallback to chars/4 estimate
    return Math.ceil(text.length / 4);
  }
}

export async function runTokenAnalysis(corpusDir, codebookPath, verbose = false) {
  const cb = new Codebook(codebookPath);
  cb.load();

  const codebookText = fs.readFileSync(codebookPath, 'utf-8');
  const codebookTokens = countTokens(codebookText);

  const files = fs.readdirSync(corpusDir)
    .filter(f => f.endsWith('.md'))
    .map(f => ({ name: f, path: path.join(corpusDir, f) }));

  const results = [];
  let totalOrigTokens = 0;
  let totalCompTokens = 0;
  let totalCompWithCBTokens = 0;

  for (const file of files) {
    const original = fs.readFileSync(file.path, 'utf-8');
    const compressed = compress(original, cb);

    // Strip header for pure content token count
    const compressedBody = compressed.replace(/^<!--.*?-->\n/, '');

    const origTokens = countTokens(original);
    const compTokens = countTokens(compressedBody);
    const compWithCBTokens = compTokens + codebookTokens;

    // Also compute chars/4 estimate for comparison
    const estOrigTokens = Math.ceil(original.length / 4);
    const estCompTokens = Math.ceil(compressedBody.length / 4);

    totalOrigTokens += origTokens;
    totalCompTokens += compTokens;
    totalCompWithCBTokens += compWithCBTokens;

    const savings = origTokens > 0 ? ((origTokens - compTokens) / origTokens * 100) : 0;
    const netSavings = origTokens > 0 ? ((origTokens - compWithCBTokens) / origTokens * 100) : 0;

    results.push({
      file: file.name,
      originalTokens: origTokens,
      compressedTokens: compTokens,
      withCodebook: compWithCBTokens,
      savings: savings.toFixed(1) + '%',
      netSavings: netSavings.toFixed(1) + '%',
      ratio: origTokens > 0 ? (origTokens / compTokens).toFixed(2) + 'x' : 'N/A',
      estimateAccuracy: estOrigTokens > 0
        ? ((1 - Math.abs(origTokens - estOrigTokens) / origTokens) * 100).toFixed(1) + '%'
        : 'N/A',
    });

    if (verbose) {
      console.log(`  ${file.name}: ${origTokens} → ${compTokens} tokens (${savings.toFixed(1)}% savings)`);
    }
  }

  const totalSavings = totalOrigTokens > 0
    ? ((totalOrigTokens - totalCompTokens) / totalOrigTokens * 100).toFixed(1)
    : 0;
  const totalNetSavings = totalOrigTokens > 0
    ? ((totalOrigTokens - totalCompWithCBTokens) / totalOrigTokens * 100).toFixed(1)
    : 0;

  return {
    name: 'Token Analysis (tiktoken cl100k_base)',
    codebookOverhead: codebookTokens + ' tokens',
    totalOriginal: totalOrigTokens,
    totalCompressed: totalCompTokens,
    totalWithCodebook: totalCompWithCBTokens,
    savings: totalSavings + '%',
    netSavings: totalNetSavings + '% (including codebook overhead)',
    ratio: totalCompTokens > 0 ? (totalOrigTokens / totalCompTokens).toFixed(2) + 'x' : 'N/A',
    details: results,
  };
}
