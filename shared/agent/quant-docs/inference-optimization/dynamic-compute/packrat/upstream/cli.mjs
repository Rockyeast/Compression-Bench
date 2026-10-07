#!/usr/bin/env node

/**
 * PackRat CLI
 * Usage:
 *   packrat init                  — Create .packrat/codebook.json
 *   packrat learn [path]          — Scan files, build/update codebook
 *   packrat compress <file>       — Compress a file → .pr.md
 *   packrat decompress <file>     — Decompress a .pr.md file
 *   packrat stats                 — Show codebook stats
 */

import { Codebook, learn, compress, decompress, compressFile, stats } from './index.mjs';
import fs from 'node:fs';

const [,, cmd, ...args] = process.argv;

const CODEBOOK_PATH = '.packrat/codebook.json';

function ensureCodebook() {
  const cb = new Codebook(CODEBOOK_PATH);
  if (!cb.load()) {
    console.error('No codebook found. Run: packrat init');
    process.exit(1);
  }
  return cb;
}

switch (cmd) {
  case 'init': {
    const cb = new Codebook(CODEBOOK_PATH);
    if (cb.load()) {
      console.log('Codebook already exists with ' + cb.size + ' entries.');
    } else {
      cb.save();
      console.log('PackRat initialized. Run: packrat learn [path]');
    }
    break;
  }

  case 'learn': {
    const cb = new Codebook(CODEBOOK_PATH);
    cb.load(); // OK if doesn't exist yet — we'll create it
    const targetDir = args[0] || '.';
    const { newEntries, filesScanned } = learn([targetDir], cb);
    console.log(`Scanned ${filesScanned} files. Learned ${newEntries} new patterns. Total: ${cb.size} entries.`);
    break;
  }

  case 'compress': {
    const file = args[0];
    if (!file) { console.error('Usage: packrat compress <file>'); process.exit(1); }
    if (!fs.existsSync(file)) { console.error('File not found: ' + file); process.exit(1); }
    const cb = ensureCodebook();
    const result = compressFile(file, cb);
    console.log(`Compressed: ${file} → ${result.outPath}`);
    console.log(`${result.originalSize} → ${result.compressedSize} chars (${result.ratio})`);
    break;
  }

  case 'decompress': {
    const file = args[0];
    if (!file) { console.error('Usage: packrat decompress <file>'); process.exit(1); }
    if (!fs.existsSync(file)) { console.error('File not found: ' + file); process.exit(1); }
    const cb = ensureCodebook();
    const content = fs.readFileSync(file, 'utf-8');
    const expanded = decompress(content, cb);
    // If the input is foo.pr.md, output is foo.expanded.md; otherwise append
    // .expanded.md to the base name. The previous chained .replace double-
    // stacked (sample.pr.md → sample.expanded.expanded.md).
    const outPath = file.endsWith('.pr.md')
      ? file.replace(/\.pr\.md$/, '.expanded.md')
      : file.replace(/\.md$/, '.expanded.md');
    fs.writeFileSync(outPath, expanded);
    console.log(`Decompressed: ${file} → ${outPath}`);
    break;
  }

  case 'stats': {
    const cb = ensureCodebook();
    const s = stats(cb);
    console.log('PackRat Codebook Stats');
    console.log('======================');
    console.log(`Version:  ${s.version}`);
    console.log(`Generated: ${s.generated}`);
    console.log(`Entities: ${s.entities}`);
    console.log(`Phrases:  ${s.phrases}`);
    console.log(`Total:    ${s.total}`);
    console.log('');
    console.log('Top entries by token savings:');
    for (const e of (s.topByTokenSavings || [])) {
      console.log(`  ${e.code} = ${e.text} (saves ${e.tokenSavings} tokens/hit)`);
    }
    break;
  }

  default:
    console.log(`PackRat — AI Memory Compression

Commands:
  packrat init              Create .packrat/codebook.json
  packrat learn [path]      Scan files, build/update codebook
  packrat compress <file>   Compress a file → .pr.md
  packrat decompress <file> Decompress a .pr.md file
  packrat stats             Show codebook stats`);
}
