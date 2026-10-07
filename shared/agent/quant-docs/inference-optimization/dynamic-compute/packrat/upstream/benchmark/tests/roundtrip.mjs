/**
 * Round-Trip Accuracy Test
 * compress → decompress, diff against original. Must be 100%.
 */
import fs from 'node:fs';
import path from 'node:path';
import { Codebook, compress, decompress } from '../../index.mjs';

export async function runRoundTrip(corpusDir, codebookPath, verbose = false) {
  const cb = new Codebook(codebookPath);
  cb.load();

  const files = fs.readdirSync(corpusDir)
    .filter(f => f.endsWith('.md'))
    .map(f => ({ name: f, path: path.join(corpusDir, f) }));

  const results = [];
  let totalChars = 0;
  let matchedChars = 0;

  for (const file of files) {
    const original = fs.readFileSync(file.path, 'utf-8');
    const compressed = compress(original, cb);
    const decompressed = decompress(compressed, cb);

    totalChars += original.length;

    // Character-level diff
    const diffs = [];
    const maxLen = Math.max(original.length, decompressed.length);
    let fileMatched = 0;

    for (let i = 0; i < maxLen; i++) {
      if (original[i] === decompressed[i]) {
        fileMatched++;
      } else {
        diffs.push({
          pos: i,
          expected: original[i] !== undefined ? JSON.stringify(original[i]) : '<EOF>',
          got: decompressed[i] !== undefined ? JSON.stringify(decompressed[i]) : '<EOF>',
          context: original.slice(Math.max(0, i - 20), i + 20),
        });
      }
    }

    matchedChars += fileMatched;
    const accuracy = maxLen > 0 ? (fileMatched / maxLen * 100) : 100;
    const pass = accuracy === 100;

    results.push({
      file: file.name,
      originalSize: original.length,
      compressedSize: compressed.length,
      decompressedSize: decompressed.length,
      accuracy: accuracy.toFixed(2) + '%',
      pass,
      diffs: diffs.slice(0, 5), // first 5 diffs only
    });

    if (verbose && !pass) {
      console.log(`  FAIL: ${file.name}`);
      for (const d of diffs.slice(0, 3)) {
        console.log(`    pos ${d.pos}: expected ${d.expected}, got ${d.got}`);
        console.log(`    context: ...${d.context}...`);
      }
    }
  }

  // Synthetic edge cases
  const edgeCases = [
    { name: 'empty-string', text: '' },
    { name: 'single-char', text: 'x' },
    { name: 'only-whitespace', text: '   \n\t\n   ' },
    { name: 'dollar-signs', text: 'export $HOME=$PATH; echo ${USER}' },
    { name: 'regex-metachars', text: 'test.*+?^${}()|[]\\end' },
    { name: 'unicode-emoji', text: 'Hello world! Testing 1-2-3' },
    { name: 'backslashes', text: 'C:\\Users\\testuser\\projects\\test' },
    { name: 'repeated-codes', text: '$K $MX $K $MX $K test $p1 $p2' },
    { name: 'newlines-only', text: '\n\n\n\n\n' },
    { name: 'very-long', text: 'x'.repeat(100000) },
  ];

  for (const tc of edgeCases) {
    const compressed = compress(tc.text, cb);
    const decompressed = decompress(compressed, cb);
    const pass = decompressed === tc.text;

    results.push({
      file: `[synthetic] ${tc.name}`,
      originalSize: tc.text.length,
      compressedSize: compressed.length,
      decompressedSize: decompressed.length,
      accuracy: pass ? '100.00%' : '0.00%',
      pass,
      diffs: pass ? [] : [{ expected: JSON.stringify(tc.text.slice(0, 50)), got: JSON.stringify(decompressed.slice(0, 50)) }],
    });
  }

  const totalPass = results.filter(r => r.pass).length;
  const totalTests = results.length;

  return {
    name: 'Round-Trip Accuracy',
    score: `${totalPass}/${totalTests}`,
    pass: totalPass === totalTests,
    accuracy: totalTests > 0 ? (totalPass / totalTests * 100).toFixed(1) + '%' : 'N/A',
    details: results,
  };
}
