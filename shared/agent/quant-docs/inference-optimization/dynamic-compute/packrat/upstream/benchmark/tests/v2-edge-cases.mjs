#!/usr/bin/env node
/**
 * PackRat v2 — Comprehensive Edge Case + Real-World Tests
 * Tests round-trip fidelity, token savings, and safety.
 * ALL TESTS ARE READ-ONLY — never modifies source files.
 */
import fs from 'node:fs';
import path from 'node:path';
import { Codebook, learn, compress, decompress, countTokens } from '../../index.mjs';

const PYTHON_PATH = process.env.PYTHON_PATH || 'python';
const results = { edge: [], stress: [], realWorld: [], summary: {} };
let totalPass = 0, totalFail = 0;

function test(category, name, input) {
  try {
    const comp = compress(input, globalCb);
    const dec = decompress(comp, globalCb);
    const pass = input === dec;
    if (pass) {
      totalPass++;
      results[category].push({ name, pass: true });
    } else {
      totalFail++;
      // Find first diff position
      let diffPos = -1;
      for (let i = 0; i < Math.max(input.length, dec.length); i++) {
        if (input[i] !== dec[i]) { diffPos = i; break; }
      }
      const ctx = 30;
      results[category].push({
        name, pass: false, diffPos,
        expected: JSON.stringify(input.slice(Math.max(0, diffPos - ctx), diffPos + ctx)),
        got: JSON.stringify(dec.slice(Math.max(0, diffPos - ctx), diffPos + ctx)),
      });
    }
    return pass;
  } catch (err) {
    totalFail++;
    results[category].push({ name, pass: false, error: err.message });
    return false;
  }
}

// ─── Load codebook ───
const cbPath = process.argv[2] || './benchmark/corpus/.packrat/codebook-v2.json';
const globalCb = new Codebook(cbPath);
if (!globalCb.load()) {
  console.error('Codebook not found at ' + cbPath);
  process.exit(1);
}
console.log('Codebook loaded: ' + globalCb.size + ' entries from ' + cbPath);
console.log('');

// ════════════════════════════════════════════
// TEST GROUP 1: EDGE CASES
// ════════════════════════════════════════════
console.log('━━━ Group 1: Edge Cases ━━━');

// Empty / whitespace
test('edge', 'empty string', '');
test('edge', 'single space', ' ');
test('edge', 'newline only', '\n');
test('edge', 'multiple newlines', '\n\n\n\n\n');
test('edge', 'tabs and spaces', '\t  \t  \t');
test('edge', 'whitespace multiline', '  \n  \n  \n');
test('edge', 'trailing newline', 'hello\n');
test('edge', 'no trailing newline', 'hello');

// Unicode
test('edge', 'emoji', 'Hello 🎉 world 🚀 PackRat 🐀 celebration 🎊');
test('edge', 'CJK', '你好世界 PackRat 压缩工具 テスト');
test('edge', 'accented chars', 'café résumé naïve über Zürich ñ');
test('edge', 'math symbols', '∑ ∫ ∂ ∇ ∞ π ε δ ≈ ≠ ≤ ≥');
test('edge', 'zero-width chars', 'test\u200B\u200Cword\u200D\uFEFF');
test('edge', 'private use area (our escape chars)', 'text with \uE000 and \uE001 embedded');

// Code blocks
test('edge', 'fenced code block', '```javascript\nfunction foo() {\n  const $x = 42;\n  return $x;\n}\n```');
test('edge', 'inline code dollar', 'Use `$HOME` or `$PATH` in shell');
test('edge', 'inline code regex', 'Match with `/\\b\\w+\\b/g` regex');
test('edge', 'nested backticks', '``code with `backtick` inside``');

// Markdown
test('edge', 'all header levels', '# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6');
test('edge', 'bold italic strike', '**bold** *italic* ***both*** ~~strike~~');
test('edge', 'links', '[PackRat](https://github.com/test) and [docs](https://docs.test.com)');
test('edge', 'markdown table', '| Col1 | Col2 | Col3 |\n|------|------|------|\n| a    | b    | c    |');
test('edge', 'blockquote nested', '> Quote\n>> Nested\n>>> Deep nested');
test('edge', 'task list', '- [x] done\n- [ ] todo\n- [x] also done');
test('edge', 'horizontal rules', '---\n***\n___');

// Strings that look like PackRat codes (CRITICAL — must survive)
test('edge', 'literal $K', 'The variable $K equals 1000');
test('edge', 'literal $P1', 'Parameter $P1 = first arg');
test('edge', 'literal $RR', 'Error $RR means retry request');
test('edge', 'dollar amounts', '$10 $20 $100 $1K $1M $1B');
test('edge', 'hash tags', '#D #T #M #P are category labels');
test('edge', 'fake codes', '$ABC $XYZ $123 $p99 $ZZZZ');
test('edge', 'mixed real + fake codes', 'The $K variable and $FAKE code');

// Special characters
test('edge', 'backslashes', 'C:\\Users\\test\\path\\to\\file');
test('edge', 'angle brackets', '<div>Hello</div> <script>alert(1)</script>');
test('edge', 'JSON', '{"key": "value", "nested": {"a": 1, "b": [2,3]}}');
test('edge', 'pipe chars', '| col1 | col2 | col3 |');
test('edge', 'regex special', '.*+?^${}()|[]\\');
test('edge', 'all printable ASCII', Array.from({length: 95}, (_, i) => String.fromCharCode(32 + i)).join(''));

// Double compression (text that already has a PackRat header)
test('edge', 'text with fake v1 header', '<!-- packrat:v1 codebook:fake -->\nSome $RR text');
test('edge', 'text with fake v2 header', '<!-- packrat:v2 codebook:fake -->\nContent after header');

const edgePass = results.edge.filter(r => r.pass).length;
const edgeFail = results.edge.filter(r => !r.pass).length;
console.log(`  ${edgePass} PASS, ${edgeFail} FAIL (${results.edge.length} tests)`);
if (edgeFail > 0) {
  for (const f of results.edge.filter(r => !r.pass)) {
    console.log(`  ❌ ${f.name}: ${f.error || 'diff at pos ' + f.diffPos}`);
    if (f.expected) console.log(`     expected: ${f.expected}`);
    if (f.got) console.log(`     got:      ${f.got}`);
  }
}
console.log('');

// ════════════════════════════════════════════
// TEST GROUP 2: STRESS TESTS
// ════════════════════════════════════════════
console.log('━━━ Group 2: Stress Tests ━━━');

test('stress', 'same word 200x', Array(200).fill('MyAppName').join(' '));
test('stress', 'same path 100x', Array(100).fill('C:/Users/devuser/projects/test/').join('\n'));
test('stress', 'same phrase 50x', Array(50).fill('via OpenRouter for free').join('. '));
test('stress', '10K char single line', 'x'.repeat(10000));
test('stress', '50K char file', ('The quick brown fox jumps. ').repeat(2000));
test('stress', 'long path', 'C:/Users/devuser/' + 'subdir/'.repeat(50) + 'file.md');
test('stress', 'long URL', 'https://example.com/' + 'path/'.repeat(50) + 'page');
test('stress', 'tiny 1-char', 'a');
test('stress', 'tiny 2-char', 'ab');
test('stress', 'only numbers', '1234567890'.repeat(100));
test('stress', 'only punctuation', '!@#$%^&*()_+-=[]{}|;:,.<>?/~`'.repeat(20));
test('stress', 'alternating entities+paths', Array(50).fill('TypeScript at C:/Users/devuser/projects/app/ uses React').join('\n'));

// Binary-ish content
test('stress', 'null bytes', 'hello\x00world\x00test');
test('stress', 'high bytes', String.fromCharCode(...Array.from({length: 50}, (_, i) => 128 + i)));

const stressPass = results.stress.filter(r => r.pass).length;
const stressFail = results.stress.filter(r => !r.pass).length;
console.log(`  ${stressPass} PASS, ${stressFail} FAIL (${results.stress.length} tests)`);
if (stressFail > 0) {
  for (const f of results.stress.filter(r => !r.pass)) {
    console.log(`  ❌ ${f.name}: ${f.error || 'diff at pos ' + f.diffPos}`);
  }
}
console.log('');

// ════════════════════════════════════════════
// TEST GROUP 3: REAL-WORLD FILES (READ-ONLY)
// ════════════════════════════════════════════
console.log('━━━ Group 3: Real-World Files (read-only) ━━━');

// First, build a codebook from real memory files (but save to temp location)
// Uses HOME env var to find memory dir — works on any machine, skips if not found
const homeDir = process.env.HOME || process.env.USERPROFILE || '';
const realMemDir = path.join(homeDir, '.claude', 'projects', 'C--Users-' + path.basename(homeDir), 'memory');
const tempCbPath = './benchmark/output/temp-real-codebook.json';
const realCb = new Codebook(tempCbPath);

if (fs.existsSync(realMemDir)) {
  console.log('  Building codebook from real memory files...');
  const learnResult = learn([realMemDir], realCb, {
    tokenAware: true,
    minFreq: 2,
    minTokenSavings: 1,
  });
  console.log(`  Learned: ${learnResult.filesScanned} files, ${realCb.size} entries (${learnResult.pruned} pruned)`);
  if (learnResult.tokenStats) {
    console.log(`  Breakdown: ${learnResult.tokenStats.pathsAdded} paths, ${learnResult.tokenStats.entitiesKept} entities, ${learnResult.tokenStats.phrasesKept} phrases`);
  }
  console.log('');

  // Now test round-trip on every real file + measure token savings
  const realFiles = fs.readdirSync(realMemDir)
    .filter(f => f.endsWith('.md') && !f.startsWith('.'));

  let totalOrigTok = 0, totalCompTok = 0, totalOrigBytes = 0, totalCompBytes = 0;
  let rtPass = 0, rtFail = 0;

  console.log('  ' + 'FILE'.padEnd(35) + 'BYTES'.padStart(7) + '  TOKENS'.padStart(7) + '  COMP_T'.padStart(7) + '  TOK_SAV'.padStart(8) + '  BYTE_SAV'.padStart(9) + '  RT');
  console.log('  ' + '-'.repeat(85));

  for (const f of realFiles) {
    const filePath = path.join(realMemDir, f);
    const original = fs.readFileSync(filePath, 'utf-8');

    // Round-trip test
    const compressed = compress(original, realCb);
    const decompressed = decompress(compressed, realCb);
    const isPass = original === decompressed;

    if (isPass) rtPass++; else rtFail++;

    // Token analysis (strip header for fair comparison)
    const compBody = compressed.replace(/^<!--\s*packrat:v\d+\s+codebook:.+?-->\n?/, '');
    const origTok = countTokens(original);
    const compTok = countTokens(compBody);

    totalOrigTok += origTok;
    totalCompTok += compTok;
    totalOrigBytes += original.length;
    totalCompBytes += compBody.length;

    const tokSav = origTok > 0 ? ((1 - compTok / origTok) * 100).toFixed(1) : '0.0';
    const byteSav = original.length > 0 ? ((1 - compBody.length / original.length) * 100).toFixed(1) : '0.0';

    const status = isPass ? '✅' : '❌';
    const shortName = f.length > 33 ? f.slice(0, 30) + '...' : f;
    console.log('  ' + shortName.padEnd(35) + String(original.length).padStart(7) + String(origTok).padStart(8) + String(compTok).padStart(8) + (tokSav + '%').padStart(8) + (byteSav + '%').padStart(9) + '  ' + status);

    results.realWorld.push({
      file: f,
      bytes: original.length,
      origTokens: origTok,
      compTokens: compTok,
      tokenSavings: tokSav + '%',
      byteSavings: byteSav + '%',
      roundTrip: isPass,
    });
  }

  console.log('  ' + '-'.repeat(85));
  const totalTokSav = totalOrigTok > 0 ? ((1 - totalCompTok / totalOrigTok) * 100).toFixed(1) : '0.0';
  const totalByteSav = totalOrigBytes > 0 ? ((1 - totalCompBytes / totalOrigBytes) * 100).toFixed(1) : '0.0';
  console.log('  ' + 'TOTAL'.padEnd(35) + String(totalOrigBytes).padStart(7) + String(totalOrigTok).padStart(8) + String(totalCompTok).padStart(8) + (totalTokSav + '%').padStart(8) + (totalByteSav + '%').padStart(9) + '  ' + (rtFail === 0 ? '✅' : '❌'));
  console.log('');
  console.log(`  Round-trip: ${rtPass} PASS, ${rtFail} FAIL`);
  console.log(`  Token savings: ${totalTokSav}% (${totalOrigTok} → ${totalCompTok}, saved ${totalOrigTok - totalCompTok} tokens)`);
  console.log(`  Byte savings: ${totalByteSav}% (${totalOrigBytes} → ${totalCompBytes}, saved ${totalOrigBytes - totalCompBytes} bytes)`);

  results.summary.realWorld = {
    files: realFiles.length,
    roundTripPass: rtPass,
    roundTripFail: rtFail,
    totalOrigTokens: totalOrigTok,
    totalCompTokens: totalCompTok,
    tokenSavings: totalTokSav + '%',
    totalOrigBytes: totalOrigBytes,
    totalCompBytes: totalCompBytes,
    byteSavings: totalByteSav + '%',
    codebookEntries: realCb.size,
  };

  // Cleanup temp codebook
  try { fs.unlinkSync(tempCbPath); } catch {}
} else {
  console.log('  ⚠️  Real memory dir not found, skipping');
}
console.log('');

// ════════════════════════════════════════════
// FINAL SUMMARY
// ════════════════════════════════════════════
console.log('╔══════════════════════════════════════════════════════╗');
console.log('║          PACKRAT v2 TEST RESULTS                    ║');
console.log('╠══════════════════════════════════════════════════════╣');
console.log(`║  Edge Cases:    ${String(edgePass).padStart(3)} pass / ${String(edgeFail).padStart(2)} fail    ${edgeFail === 0 ? '✅' : '❌'}             ║`);
console.log(`║  Stress Tests:  ${String(stressPass).padStart(3)} pass / ${String(stressFail).padStart(2)} fail    ${stressFail === 0 ? '✅' : '❌'}             ║`);
if (results.summary.realWorld) {
  const rw = results.summary.realWorld;
  console.log(`║  Real-World:    ${String(rw.roundTripPass).padStart(3)} pass / ${String(rw.roundTripFail).padStart(2)} fail    ${rw.roundTripFail === 0 ? '✅' : '❌'}             ║`);
  console.log(`║  Token Savings: ${rw.tokenSavings.padStart(7)}                            ║`);
  console.log(`║  Byte Savings:  ${rw.byteSavings.padStart(7)}                            ║`);
}
console.log(`║  Total:         ${String(totalPass).padStart(3)} pass / ${String(totalFail).padStart(2)} fail    ${totalFail === 0 ? '✅ ALL CLEAR' : '❌ FIX NEEDED'}     ║`);
console.log('╚══════════════════════════════════════════════════════╝');

// Save full results to file
const reportPath = './benchmark/output/v2-test-results.json';
fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
console.log('');
console.log('Full results saved to: ' + reportPath);

// Exit with error code if any failures
if (totalFail > 0) process.exit(1);
