"use strict";

// The core primitive shared by every encoder: factor out the redundant,
// repeated parts of a column of strings, keeping only what varies.
//
// Given a list of strings that all share a common prefix and/or suffix
// (e.g. "reqId=1", "reqId=2", ... share prefix "reqId="), we record the
// prefix/suffix ONCE and keep only each value's varying middle. This is
// fully reversible: value === prefix + middle + suffix.

function longestCommonPrefix(strings) {
  if (strings.length === 0) return "";
  let prefix = strings[0];
  for (let i = 1; i < strings.length && prefix; i++) {
    const s = strings[i];
    let j = 0;
    const max = Math.min(prefix.length, s.length);
    while (j < max && prefix[j] === s[j]) j++;
    prefix = prefix.slice(0, j);
  }
  return prefix;
}

function longestCommonSuffix(strings) {
  if (strings.length === 0) return "";
  let suffix = strings[0];
  for (let i = 1; i < strings.length && suffix; i++) {
    const s = strings[i];
    let j = 0;
    const max = Math.min(suffix.length, s.length);
    while (j < max && suffix[suffix.length - 1 - j] === s[s.length - 1 - j]) j++;
    suffix = suffix.slice(suffix.length - j);
  }
  return suffix;
}

// Factor a column of values into { prefix, suffix, middles }.
// Guarantees: prefix + middles[i] + suffix === values[i] for every i.
// Prefix and suffix are chosen to NOT overlap (important for short values).
function factorColumn(values) {
  if (values.length === 0) return { prefix: "", suffix: "", middles: [] };

  let prefix = longestCommonPrefix(values);
  // Suffix is computed on what's left after removing the prefix, so prefix and
  // suffix can never overlap on a short value like "reqId=" itself.
  const afterPrefix = values.map((v) => v.slice(prefix.length));
  let suffix = longestCommonSuffix(afterPrefix);

  const middles = afterPrefix.map((v) => v.slice(0, v.length - suffix.length));

  // Verify the invariant; if anything is off, factor nothing (safety).
  for (let i = 0; i < values.length; i++) {
    if (prefix + middles[i] + suffix !== values[i]) {
      return { prefix: "", suffix: "", middles: values.slice() };
    }
  }
  return { prefix, suffix, middles };
}

module.exports = { longestCommonPrefix, longestCommonSuffix, factorColumn };
