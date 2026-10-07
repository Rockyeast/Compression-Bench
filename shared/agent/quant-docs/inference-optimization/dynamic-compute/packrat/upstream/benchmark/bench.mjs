#!/usr/bin/env node
/**
 * PackRat Benchmark Suite — "Smoke the Benchmarks" Edition
 *
 * Usage: node benchmark/bench.mjs [--corpus <path>] [--codebook <path>] [--verbose]
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { runRoundTrip } from './tests/roundtrip.mjs';
import { runTokenAnalysis } from './tests/tokens.mjs';
import { runCodebookEfficiency } from './tests/codebook-efficiency.mjs';
import { runTraversalBenchmark } from './tests/traversal.mjs';
import { runBreakEven } from './tests/breakeven.mjs';
import { runLLMReadability } from './tests/llm-readability.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Parse CLI args
const args = process.argv.slice(2);
const verbose = args.includes('--verbose') || args.includes('-v');
const corpusIdx = args.indexOf('--corpus');
const cbIdx = args.indexOf('--codebook');

const corpusDir = corpusIdx >= 0 ? args[corpusIdx + 1] : path.join(__dirname, 'corpus');
const codebookPath = cbIdx >= 0 ? args[cbIdx + 1] : path.join(corpusDir, '.packrat', 'codebook.json');
const outputDir = path.join(__dirname, 'output');

if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

console.log('');
console.log('╔══════════════════════════════════════════════════╗');
console.log('║       PACKRAT BENCHMARK SUITE v1.0               ║');
console.log('║       "Smoke the Benchmarks" Edition              ║');
console.log('╚══════════════════════════════════════════════════╝');
console.log('');
console.log(`Corpus:   ${corpusDir}`);
console.log(`Codebook: ${codebookPath}`);
console.log(`Verbose:  ${verbose}`);
console.log('');

const allResults = {};

// ── Test 1: Round-Trip Accuracy ──
console.log('━━━ Test 1: Round-Trip Accuracy ━━━');
const rt = await runRoundTrip(corpusDir, codebookPath, verbose);
allResults.roundTrip = rt;
console.log(`  Result: ${rt.accuracy} (${rt.score})`);
console.log(`  Status: ${rt.pass ? '✅ PASS' : '❌ FAIL'}`);
if (!rt.pass) {
  const failures = rt.details.filter(d => !d.pass);
  for (const f of failures) {
    console.log(`    FAIL: ${f.file} — ${f.accuracy}`);
    if (f.diffs.length > 0) {
      console.log(`      First diff: ${JSON.stringify(f.diffs[0])}`);
    }
  }
}
console.log('');

// ── Test 2: Token Analysis ──
console.log('━━━ Test 2: Token Analysis (tiktoken) ━━━');
const tk = await runTokenAnalysis(corpusDir, codebookPath, verbose);
allResults.tokens = tk;
console.log(`  Original:   ${tk.totalOriginal} tokens`);
console.log(`  Compressed: ${tk.totalCompressed} tokens`);
console.log(`  Savings:    ${tk.savings} (${tk.ratio} compression)`);
console.log(`  Net savings (w/ codebook): ${tk.netSavings}`);
console.log('');

// ── Test 3: Codebook Efficiency ──
console.log('━━━ Test 3: Codebook Efficiency ━━━');
const ce = await runCodebookEfficiency(corpusDir, codebookPath, verbose);
allResults.codebookEfficiency = ce;
console.log(`  Entries: ${ce.totalEntries} total, ${ce.activeEntries} active, ${ce.deadEntries} dead`);
console.log(`  Utilization: ${ce.utilization}`);
console.log(`  Total chars saved: ${ce.totalCharsSaved}`);
console.log(`  Collisions: ${ce.collisions}`);
console.log('');

// ── Test 4: Traversal Speed ──
console.log('━━━ Test 4: Traversal Speed ━━━');
const tv = await runTraversalBenchmark(corpusDir, codebookPath, verbose);
allResults.traversal = tv;
console.log(`  Raw corpus: ${tv.rawCorpusSize}`);
console.log(`  Compressed: ${tv.compressedCorpusSize} (${tv.sizeReduction} smaller)`);
console.log(`  Raw search: ${tv.rawSearchMedian}`);
console.log(`  Compressed search: ${tv.compressedSearchMedian}`);
console.log(`  Speedup: ${tv.speedup}`);
console.log('');

// ── Test 5: Break-Even Analysis ──
console.log('━━━ Test 5: Break-Even Analysis ━━━');
const be = await runBreakEven(codebookPath, verbose);
allResults.breakEven = be;
console.log(`  Codebook overhead: ${be.codebookSize}`);
console.log(`  Break-even point: ${be.breakEvenPoint}`);
console.log(be.chart);
console.log('');

// ── Test 6: LLM Readability ──
console.log('━━━ Test 6: LLM Readability (prompts generated) ━━━');
const lr = await runLLMReadability(corpusDir, codebookPath, outputDir, verbose);
allResults.llmReadability = lr;
console.log(`  Test sets: ${lr.testSets.length} files, ${lr.totalQuestions} questions`);
for (const ts of lr.testSets) {
  console.log(`    ${ts.file}: ${ts.questions} Qs, ${ts.tokenSavings} token savings`);
}
console.log(`  Prompt files saved to: ${outputDir}`);
console.log('');

// ── Generate Report ──
const report = generateReport(allResults);
const reportPath = path.join(__dirname, 'RESULTS.md');
fs.writeFileSync(reportPath, report);
console.log(`Report saved to: ${reportPath}`);
console.log('');

// ── Score Card ──
console.log('╔══════════════════════════════════════════════════╗');
console.log('║              PACKRAT SCORE CARD                  ║');
console.log('╠══════════════════════════════════════════════════╣');
console.log(`║  Round-Trip Accuracy:  ${rt.accuracy.padEnd(27)}║`);
console.log(`║  Token Savings:        ${(tk.savings + ' (' + tk.ratio + ')').padEnd(27)}║`);
console.log(`║  Codebook Utilization: ${ce.utilization.padEnd(27)}║`);
console.log(`║  Traversal Speedup:    ${tv.speedup.padEnd(27)}║`);
console.log(`║  Break-Even Point:     ${be.breakEvenPoint.padEnd(27)}║`);
console.log(`║  LLM Test Questions:   ${(lr.totalQuestions + ' generated').padEnd(27)}║`);
console.log('╠══════════════════════════════════════════════════╣');
const verdict = rt.pass ? '✅ PERFECT ROUND-TRIP' : '❌ ROUND-TRIP FAILURES';
console.log(`║  Verdict: ${verdict.padEnd(39)}║`);
console.log('╚══════════════════════════════════════════════════╝');
console.log('');

function generateReport(results) {
  const now = new Date().toISOString();
  const r = results;

  let md = `# PackRat Benchmark Results\n\n`;
  md += `**Date:** ${now}\n`;
  md += `**Corpus:** ${corpusDir}\n`;
  md += `**Codebook:** ${codebookPath}\n\n`;

  // Score card
  md += `## Score Card\n\n`;
  md += `| Metric | Result |\n|--------|--------|\n`;
  md += `| Round-Trip Accuracy | ${r.roundTrip.accuracy} (${r.roundTrip.score}) |\n`;
  md += `| Token Savings | ${r.tokens.savings} (${r.tokens.ratio}) |\n`;
  md += `| Net Savings (w/ codebook) | ${r.tokens.netSavings} |\n`;
  md += `| Codebook Utilization | ${r.codebookEfficiency.utilization} (${r.codebookEfficiency.activeEntries}/${r.codebookEfficiency.totalEntries}) |\n`;
  md += `| Traversal Speedup | ${r.traversal.speedup} |\n`;
  md += `| Break-Even Point | ${r.breakEven.breakEvenPoint} |\n`;
  md += `| LLM Test Questions | ${r.llmReadability.totalQuestions} generated |\n\n`;

  // Round-trip details
  md += `## 1. Round-Trip Accuracy\n\n`;
  md += `| File | Original | Compressed | Accuracy | Status |\n|------|----------|------------|----------|--------|\n`;
  for (const d of r.roundTrip.details) {
    md += `| ${d.file} | ${d.originalSize}B | ${d.compressedSize}B | ${d.accuracy} | ${d.pass ? 'PASS' : 'FAIL'} |\n`;
  }
  md += `\n`;

  // Token analysis
  md += `## 2. Token Analysis (tiktoken cl100k_base)\n\n`;
  md += `Codebook overhead: ${r.tokens.codebookOverhead}\n\n`;
  md += `| File | Original | Compressed | Savings | Ratio |\n|------|----------|------------|---------|-------|\n`;
  for (const d of r.tokens.details) {
    md += `| ${d.file} | ${d.originalTokens} | ${d.compressedTokens} | ${d.savings} | ${d.ratio} |\n`;
  }
  md += `\n**Total:** ${r.tokens.totalOriginal} → ${r.tokens.totalCompressed} tokens (${r.tokens.savings} savings, ${r.tokens.ratio})\n\n`;

  // Codebook efficiency
  md += `## 3. Codebook Efficiency\n\n`;
  md += `- Active entries: ${r.codebookEfficiency.activeEntries}/${r.codebookEfficiency.totalEntries}\n`;
  md += `- Dead entries: ${r.codebookEfficiency.deadEntries}\n`;
  md += `- Total chars saved: ${r.codebookEfficiency.totalCharsSaved}\n`;
  md += `- Code collisions: ${r.codebookEfficiency.collisions}\n\n`;
  md += `**Top performers:**\n\n`;
  md += `| Code | Text | Hits | Chars Saved |\n|------|------|------|-------------|\n`;
  for (const t of r.codebookEfficiency.topPerformers) {
    md += `| ${t.code} | ${t.text} | ${t.hits} | ${t.saved} |\n`;
  }
  md += `\n`;

  // Traversal
  md += `## 4. Traversal Speed\n\n`;
  md += `- Raw corpus: ${r.traversal.rawCorpusSize}\n`;
  md += `- Compressed: ${r.traversal.compressedCorpusSize} (${r.traversal.sizeReduction} smaller)\n`;
  md += `- Raw search median: ${r.traversal.rawSearchMedian}\n`;
  md += `- Compressed search median: ${r.traversal.compressedSearchMedian}\n`;
  md += `- **Speedup: ${r.traversal.speedup}**\n\n`;

  // Break-even
  md += `## 5. Break-Even Analysis\n\n`;
  md += `Break-even point: **${r.breakEven.breakEvenPoint}**\n\n`;
  md += `\`\`\`\n${r.breakEven.chart}\n\`\`\`\n\n`;

  // LLM readability
  md += `## 6. LLM Readability\n\n`;
  md += `Prompt files generated for manual testing:\n\n`;
  for (const ts of r.llmReadability.testSets) {
    md += `- **${ts.file}**: ${ts.questions} questions, ${ts.tokenSavings} token savings\n`;
  }
  md += `\n${r.llmReadability.instructions.join('\n')}\n`;

  return md;
}
