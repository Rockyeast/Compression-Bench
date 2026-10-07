/**
 * Break-Even Analysis
 * Finds the minimum file size where compression saves space (including codebook overhead).
 */
import fs from 'node:fs';
import { Codebook, compress } from '../../index.mjs';

export async function runBreakEven(codebookPath, verbose = false) {
  const cb = new Codebook(codebookPath);
  cb.load();

  const codebookText = fs.readFileSync(codebookPath, 'utf-8');
  const codebookSize = codebookText.length;
  const headerSize = `<!-- packrat:v1 codebook:${codebookPath} -->\n`.length;

  // Generate test texts at various sizes using realistic content
  const sampleWords = [
    'TypeScript', 'React', 'Supabase', 'memory', 'compression', 'system',
    'PackRat', 'auto-learns', 'codebooks', 'from', 'files', 'and', 'compresses',
    'GitHub', 'Telegram', 'Docker', 'Vercel', 'Cloudflare', 'FastAPI',
    'The', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'with',
    'API', 'tokens', 'context', 'window', 'agent', 'deployment',
    'Python', 'JavaScript', 'Node.js', 'database', 'migration', 'auth',
  ];

  function generateText(targetSize) {
    let text = '';
    let i = 0;
    while (text.length < targetSize) {
      text += sampleWords[i % sampleWords.length] + ' ';
      i++;
    }
    return text.slice(0, targetSize);
  }

  const sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000];
  const dataPoints = [];
  let breakEvenPoint = null;

  for (const size of sizes) {
    const text = generateText(size);
    const compressed = compress(text, cb);
    const compressedBody = compressed.length; // includes header
    const withCodebook = compressedBody + codebookSize;

    const netSavings = size - compressedBody;
    const netSavingsWithCB = size - withCodebook;
    const worthIt = netSavings > 0;
    const worthItWithCB = netSavingsWithCB > 0;

    if (worthIt && !breakEvenPoint) {
      breakEvenPoint = size;
    }

    dataPoints.push({
      inputSize: size,
      compressedSize: compressedBody,
      withCodebook: withCodebook,
      savings: netSavings,
      savingsWithCB: netSavingsWithCB,
      ratio: (size / compressedBody).toFixed(2) + 'x',
      worthIt,
      worthItWithCB,
    });

    if (verbose) {
      const indicator = worthIt ? '+' : '-';
      console.log(`  ${indicator} ${size} bytes → ${compressedBody} bytes (${netSavings > 0 ? 'saves' : 'costs'} ${Math.abs(netSavings)} bytes)`);
    }
  }

  // Generate ASCII chart
  const chart = generateChart(dataPoints);

  return {
    name: 'Break-Even Analysis',
    codebookSize: codebookSize + ' bytes',
    headerSize: headerSize + ' bytes',
    breakEvenPoint: breakEvenPoint ? breakEvenPoint + ' bytes' : 'Not found in range',
    dataPoints,
    chart,
    note: 'Break-even = where compressed output is smaller than original (excluding codebook, which is loaded once).',
  };
}

function generateChart(points) {
  const maxSize = Math.max(...points.map(p => p.inputSize));
  const maxSavings = Math.max(...points.map(p => p.savings));
  const minSavings = Math.min(...points.map(p => p.savings));
  const range = maxSavings - minSavings || 1;

  const width = 50;
  const lines = ['', '  Savings (bytes) vs Input Size', '  ' + '-'.repeat(width + 2)];

  for (const p of points) {
    const barLen = Math.round(((p.savings - minSavings) / range) * width);
    const bar = p.savings >= 0
      ? ' '.repeat(Math.max(0, Math.round((-minSavings / range) * width))) + '#'.repeat(Math.max(1, barLen - Math.round((-minSavings / range) * width)))
      : '#'.repeat(Math.max(1, barLen));
    const label = `${String(p.inputSize).padStart(6)} | `;
    const value = ` ${p.savings > 0 ? '+' : ''}${p.savings}`;
    lines.push('  ' + label + bar.slice(0, width) + value);
  }

  lines.push('  ' + '-'.repeat(width + 2));
  lines.push('  # = bytes saved (positive = compression wins)');
  return lines.join('\n');
}
