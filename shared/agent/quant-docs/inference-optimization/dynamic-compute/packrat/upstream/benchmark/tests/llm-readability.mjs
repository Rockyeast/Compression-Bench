/**
 * LLM Readability Test
 * Generates paired prompts (raw vs compressed) with factual questions.
 * Outputs prompt files for testing with any LLM.
 */
import fs from 'node:fs';
import path from 'node:path';
import { Codebook, compress } from '../../index.mjs';

// Hardcoded factual questions for specific corpus files
const QUESTION_SETS = {
  'sample-project.md': [
    'What database is used in Project Alpha?',
    'What language is Project Beta written in?',
    'What deployment platform replaced Netlify?',
    'What framework is used for Project Beta\'s API?',
    'Why was TypeScript chosen over JavaScript?',
  ],
  'sample-tasks.md': [
    'What is blocked waiting on Cloudflare support?',
    'How many files were migrated from JavaScript to TypeScript?',
    'What testing framework is used for React?',
    'What is the deployment platform for preview deploys?',
    'What bot is used for deploy notifications?',
  ],
};

export async function runLLMReadability(corpusDir, codebookPath, outputDir, verbose = false) {
  const cb = new Codebook(codebookPath);
  cb.load();

  const codebookText = fs.readFileSync(codebookPath, 'utf-8');

  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

  const testSets = [];

  for (const [fileName, questions] of Object.entries(QUESTION_SETS)) {
    const filePath = path.join(corpusDir, fileName);
    if (!fs.existsSync(filePath)) continue;

    const original = fs.readFileSync(filePath, 'utf-8');
    const compressed = compress(original, cb);

    // Generate raw prompt
    const rawPrompt = [
      'Read the following document and answer each question with a short, factual answer.',
      '',
      '---BEGIN DOCUMENT---',
      original,
      '---END DOCUMENT---',
      '',
      'Questions:',
      ...questions.map((q, i) => `${i + 1}. ${q}`),
      '',
      'Answer each question with just the fact, no explanation needed.',
    ].join('\n');

    // Generate compressed prompt
    const compressedPrompt = [
      'This document uses PackRat compression. The codebook below maps short codes to their full text.',
      'Read the codebook first, then read the compressed document, and answer each question.',
      '',
      '---BEGIN CODEBOOK---',
      codebookText,
      '---END CODEBOOK---',
      '',
      '---BEGIN COMPRESSED DOCUMENT---',
      compressed,
      '---END COMPRESSED DOCUMENT---',
      '',
      'Questions:',
      ...questions.map((q, i) => `${i + 1}. ${q}`),
      '',
      'Answer each question with just the fact, no explanation needed.',
    ].join('\n');

    const baseName = fileName.replace('.md', '');
    const rawPath = path.join(outputDir, `${baseName}-raw-prompt.txt`);
    const compPath = path.join(outputDir, `${baseName}-compressed-prompt.txt`);

    fs.writeFileSync(rawPath, rawPrompt);
    fs.writeFileSync(compPath, compressedPrompt);

    testSets.push({
      file: fileName,
      questions: questions.length,
      rawPromptTokensEst: Math.ceil(rawPrompt.length / 4),
      compressedPromptTokensEst: Math.ceil(compressedPrompt.length / 4),
      tokenSavings: ((1 - compressedPrompt.length / rawPrompt.length) * 100).toFixed(1) + '%',
      rawPromptPath: rawPath,
      compressedPromptPath: compPath,
    });

    if (verbose) {
      console.log(`  ${fileName}: ${questions.length} questions`);
      console.log(`    Raw prompt: ~${Math.ceil(rawPrompt.length / 4)} tokens`);
      console.log(`    Compressed prompt: ~${Math.ceil(compressedPrompt.length / 4)} tokens`);
    }
  }

  return {
    name: 'LLM Readability',
    testSets,
    totalQuestions: testSets.reduce((s, t) => s + t.questions, 0),
    outputDir,
    instructions: [
      'To complete this benchmark:',
      '1. Send each *-raw-prompt.txt to an LLM and record answers',
      '2. Send each *-compressed-prompt.txt to the SAME LLM and record answers',
      '3. Compare: if compressed answers match raw answers, readability = 100%',
      'Prompt files are saved in: ' + outputDir,
    ],
  };
}
