"use strict";

// ctxfold — lossless, structure-aware re-encoding of bulky text for LLM
// prompts. One rule above all: LOSSLESS OR NO-OP. Every encoder ships a decoder,
// and compress() verifies decode(encode(x)) === x exactly before returning the
// compressed form. If the check fails, the original text is returned untouched.
// The model reads the compressed text directly — there is no decode step at
// runtime. The decoder exists only to PROVE the transform loses nothing.

const logs = require("./encoders/logs");
const json = require("./encoders/json");
const csv = require("./encoders/csv");

// Encoders are tried in priority order.
const ENCODERS = [json, csv, logs];

// Default losslessness check: exact byte match. Encoders may override with their
// own .verify(original, restored) — e.g. JSON compares parsed data, not bytes.
function verifyDefault(original, restored) {
  return original === restored;
}

// Cheap, dependency-free token estimate. For exact counts, pass opts.countTokens
// (e.g. a tiktoken/gpt-tokenizer function); the benchmark does this.
function estimateTokens(s) {
  return Math.ceil(s.length / 4);
}

function compress(text, opts = {}) {
  if (typeof text !== "string") throw new TypeError("compress(text): text must be a string");
  const count = opts.countTokens || estimateTokens;
  const only = opts.only ? new Set([].concat(opts.only)) : null;

  const base = {
    encoder: "none",
    lossless: true,
    charsBefore: text.length,
    charsAfter: text.length,
    tokensBefore: count(text),
    tokensAfter: count(text),
  };

  for (const enc of ENCODERS) {
    if (only && !only.has(enc.name)) continue;
    let result;
    try {
      if (!enc.detect(text)) continue;
      result = enc.encode(text, opts);
    } catch {
      continue;
    }
    if (!result || !result.ok) continue;

    // The safety net: must reproduce the input (per the encoder's own
    // definition of lossless), or we don't use it.
    let restored;
    try {
      restored = enc.decode(result.encoded);
    } catch {
      continue;
    }
    const verify = enc.verify || verifyDefault;
    if (!verify(text, restored)) continue;

    // Must actually save something, or it's not worth the legend overhead.
    if (result.encoded.length >= text.length) continue;

    return {
      text: result.encoded,
      stats: {
        encoder: enc.name,
        lossless: true,
        charsBefore: text.length,
        charsAfter: result.encoded.length,
        charRatio: 1 - result.encoded.length / text.length,
        tokensBefore: count(text),
        tokensAfter: count(result.encoded),
        tokenRatio: 1 - count(result.encoded) / count(text),
      },
    };
  }

  return { text, stats: base };
}

// Round-trip a compressed string back to the original (for verification or for
// any caller that wants the data back programmatically).
//
// The magic prefix identifies the encoder unambiguously, so once it matches, a
// decode failure is a real error and is raised. It used to be swallowed by a
// "try the next encoder" catch, which meant a corrupt or newer-format payload
// came back as its own folded TEXT — indistinguishable, to the caller, from a
// string that was never a ctxfold payload at all. Use validate() for the
// non-throwing check; that is what it is for.
function decompress(encoded) {
  if (typeof encoded !== "string") return encoded;
  const enc = ENCODERS.find((e) => encoded.startsWith("\u27e6cf/" + e.name));
  if (!enc) return encoded; // not a ctxfold payload; return as-is
  return enc.decode(encoded);
}

// Check that a folded payload is a well-formed, self-consistent ctxfold output:
// its schema and rows agree and it decodes cleanly. Catches drift such as a row
// whose column count no longer matches the header, a truncated rows section, or
// an out-of-range dictionary code.
//
// Scope: validate() can confirm a payload is structurally sound and decodable.
// It CANNOT confirm a payload faithfully represents some original source it never
// saw — that's only guaranteed for payloads ctxfold produced itself (compress()
// self-checks at encode time). Returns { valid, encoder?, reason? }.
function validate(payload) {
  if (typeof payload !== "string") return { valid: false, reason: "input is not a string" };
  const enc = ENCODERS.find((e) => payload.startsWith("\u27e6cf/" + e.name));
  if (!enc) return { valid: false, reason: "not a ctxfold payload (no cf/ header)" };
  try {
    enc.decode(payload);
    return { valid: true, encoder: enc.name };
  } catch (e) {
    return { valid: false, encoder: enc.name, reason: e.message };
  }
}

module.exports = { compress, decompress, validate, estimateTokens, ENCODERS };

// Late-bound to avoid a circular require at load time (profile.js requires
// this module for compress/estimateTokens).
const { profile, renderProfile } = require("./profile");
module.exports.profile = profile;
module.exports.renderProfile = renderProfile;
