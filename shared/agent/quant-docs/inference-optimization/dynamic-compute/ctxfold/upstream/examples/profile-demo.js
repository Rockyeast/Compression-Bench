"use strict";

// Profiler demo — the one example that needs NO API key and no setup:
//
//   node examples/profile-demo.js
//
// Generates two realistic inputs (a paginated JSON API response and a server
// log), profiles both, and prints the reports. Deterministic: same output
// every run. This is also the source of the example output shown in the
// README, so docs and behavior can't drift apart.

const { profile, renderProfile } = require("../src/index");

// --- a realistic paginated API response (wrapped array, pretty-printed) ----

function makeApiResponse(n = 300) {
  const plans = ["free", "pro", "team"];
  const regions = ["us-east-1", "eu-west-1", "ap-south-1"];
  const users = [];
  for (let i = 0; i < n; i++) {
    users.push({
      id: "usr_" + String(100000 + i),
      email: "user" + i + "@example.com",
      plan: plans[i % 3],
      region: regions[i % 3],
      created_at: "2026-0" + (1 + (i % 6)) + "-" + String(1 + (i % 28)).padStart(2, "0") + "T10:00:00Z",
      active: i % 7 !== 0,
      login_count: (i * 13) % 500,
    });
  }
  return JSON.stringify({ results: users, page: 1, total: n }, null, 2);
}

// --- a realistic server log ------------------------------------------------

function makeServerLog(n = 500) {
  const paths = ["/v1/users", "/v1/items", "/v1/orders", "/health"];
  const lines = [];
  for (let i = 0; i < n; i++) {
    lines.push(
      "2026-07-18T" + String(8 + (i % 12)).padStart(2, "0") + ":" +
      String(i % 60).padStart(2, "0") + ":" + String((i * 7) % 60).padStart(2, "0") + "Z " +
      (i % 20 === 0 ? "WARN" : "INFO") + " [api] reqId=" + (40000 + i) +
      " handled request path=" + paths[i % 4] +
      " status=" + (i % 25 === 0 ? 500 : 200) + " ms=" + (5 + ((i * 17) % 300))
    );
  }
  return lines.join("\n") + "\n";
}

// ---------------------------------------------------------------------------

console.log("=== JSON API response ===\n");
console.log(renderProfile(profile(makeApiResponse())));

console.log("=== server log ===\n");
console.log(renderProfile(profile(makeServerLog())));

console.log("For exact token counts instead of estimates:");
console.log("  npm i -D gpt-tokenizer");
console.log('  then pass { countTokens } to profile() — see README.');
