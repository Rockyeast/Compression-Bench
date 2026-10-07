/* ctxfold — live profiler for the landing page.
 *
 * Drives the existing .ruler and .report components from real profile()
 * output. If this script fails to load, or the bundle is missing, the
 * static markup in index.html stays on screen unchanged.
 *
 * The report is produced by colourising renderProfile() rather than
 * re-rendering from the object, so the page and the CLI can never drift.
 */
(function () {
  "use strict";

  var IDLE = "runs entirely in your browser — no upload, no API key";

  var cf = window.ctxfold;
  if (!cf || typeof cf.profile !== "function") return;

  var input = document.getElementById("pf-input");
  var report = document.getElementById("pf-report");
  var bar = document.getElementById("pf-bar");
  var marks = document.getElementById("pf-marks");
  var note = document.getElementById("pf-note");
  var verdict = document.getElementById("pf-verdict");
  var status = document.getElementById("pf-status");
  var wrap = document.getElementById("pf-tryit");
  var ruler = document.getElementById("pf-ruler");
  var hint = document.getElementById("pf-hint");
  var run = document.getElementById("pf-run");
  if (!input || !report || !bar || !marks || !note || !run) return;

  /* ---------------------------------------------------------------- utils */

  // Results start empty and stay empty until Profile is pressed. The static
  // example lives in its own block in index.html and is never touched here,
  // so nothing on screen can be mistaken for a measurement of your input.
  function reset() {
    if (ruler) ruler.hidden = true;
    report.hidden = true;
    if (hint) hint.hidden = false;
    bar.innerHTML = "";
    marks.innerHTML = "";
    note.innerHTML = "";
    if (verdict) verdict.innerHTML = "";
    if (status) status.textContent = IDLE;
  }

  function show() {
    if (hint) hint.hidden = true;
    if (ruler) ruler.hidden = false;
    report.hidden = false;
  }

  // Composition labels and affixes echo the user's own data
  // (e.g. `constant columns  46%  c, d`). Escape before any innerHTML.
  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function pct(x) {
    return Math.round(x * 100) + "%";
  }

  function commas(n) {
    return n.toLocaleString("en-US");
  }

  /* ------------------------------------------------------------- the bar */

  // Segment count and labels vary by format: JSON and CSV give four,
  // logs give two. Cycle the existing palette instead of the fixed
  // .seg-keys/.seg-syn/.seg-val/.seg-ws classes, which are JSON-only.
  var PALETTE = [
    "var(--marker)",
    "var(--marker-soft)",
    "var(--paper)",
    "var(--blue-soft)",
  ];

  function renderRuler(p) {
    var comp = p.composition || [];

    if (!comp.length) {
      bar.innerHTML =
        '<div style="width:100%;background:var(--grid)">no composition</div>';
      marks.innerHTML = "";
      note.textContent = p.detail || "nothing to break down";
      return;
    }

    bar.innerHTML = comp
      .map(function (c, i) {
        var w = (c.pct * 100).toFixed(4) + "%";
        var fill = PALETTE[i % PALETTE.length];
        // Below ~6% there is no room for the number inside the segment.
        var label = c.pct >= 0.06 ? pct(c.pct) : "";
        return (
          '<div style="width:' +
          w +
          ";background:" +
          fill +
          '" title="' +
          esc(c.label + " — " + pct(c.pct) + " — " + c.note) +
          '">' +
          label +
          "</div>"
        );
      })
      .join("");

    marks.innerHTML = comp
      .map(function (c) {
        return (
          '<span style="width:' +
          (c.pct * 100).toFixed(4) +
          '%" title="' +
          esc(c.note) +
          '">' +
          esc(c.label) +
          "</span>"
        );
      })
      .join("");

    var n = esc(p.detail || p.format) + " — " + commas(p.chars) + " characters";
    if (p.compositionNormalized) n += " (normalised to 100%)";
    note.innerHTML = n;
  }

  /* -------------------------------------------------------- the verdict */

  // The savings number is the payoff and was previously only reachable by
  // scrolling into the report. Headline it — but take the library's own
  // judgement, so CSV (foldable yet not readable) never shows a green number.
  function renderVerdict(p) {
    if (!verdict) return;

    var text = p.verdict || "";
    var best = p.foldable && p.foldable.length ? p.foldable[0] : null;
    var willFold = /^fold it/.test(text);

    var head;
    if (willFold && best) head = pct(best.tokenRatio) + " fewer tokens";
    else if (best) head = "not worth folding to read";
    else head = "nothing to fold";

    var sub =
      willFold && best
        ? commas(p.tokens) + " → ~" + commas(best.tokensAfter) + " · lossless, verified by round-trip"
        : text;

    verdict.className = willFold ? "verdict" : "verdict warn";
    verdict.innerHTML =
      '<span class="verdict-num">' + esc(head) + "</span>" +
      '<span class="verdict-sub">' + esc(sub) + "</span>";
  }

  /* ---------------------------------------------------------- the report */

  // Section-aware colouring. Percentages mean different things in the
  // composition block (.y) and the foldable block (.b), so the pass
  // tracks which block it is inside rather than matching blindly.
  function colorize(text) {
    var section = "head";

    return text
      .split("\n")
      .map(function (line) {
        var safe = esc(line);

        if (/^where the characters go/.test(line)) section = "comp";
        else if (/^foldable \(/.test(line)) section = "fold";
        else if (/^why nothing folded/.test(line)) section = "why";
        else if (/^verdict:/.test(line)) section = "verdict";

        if (section === "head") {
          return safe.replace(
            /(\(estimated;[^)]*\))/,
            '<span class="g">$1</span>'
          );
        }

        if (section === "comp") {
          return safe.replace(/(\d+%)/, '<span class="y">$1</span>');
        }

        if (section === "fold") {
          safe = safe.replace(/(-\d+%)/, '<span class="b">$1</span>');
          // The published CSV negative result — flag it, do not bury it.
          return safe.replace(
            /(pipeline-only[^<]*)$/,
            '<span class="y">$1</span>'
          );
        }

        if (section === "why") {
          return safe.replace(/^(\s+- .*)$/, '<span class="y">$1</span>');
        }

        if (section === "verdict") {
          var m = safe.match(/^verdict:\s*(.*)$/);
          if (!m) return safe;
          var cls = /^fold it/.test(m[1]) ? "g" : "y";
          return 'verdict: <span class="' + cls + '">' + m[1] + "</span>";
        }

        return safe;
      })
      .join("\n");
  }

  /* ------------------------------------------------------------ the loop */

  function render(text) {
    if (!text.trim()) {
      reset();
      return;
    }

    show();
    var t0 = performance.now();
    var p;
    try {
      p = cf.profile(text);
    } catch (e) {
      // profile() is not supposed to throw on any input. If it does,
      // that is a bug worth seeing rather than swallowing.
      report.innerHTML =
        '<span class="y">profiler error: ' + esc(e.message) + "</span>";
      status.textContent = "unexpected error — please open an issue";
      return;
    }
    var ms = Math.round(performance.now() - t0);

    renderVerdict(p);
    renderRuler(p);
    report.innerHTML = colorize(cf.renderProfile(p));
    status.textContent =
      commas(text.length) + " characters profiled in " + ms + "ms — in your browser, nothing uploaded";

    // The results render below the button that produced them, so on a short
    // viewport nothing visibly happens. Put the headline in view instead of
    // leaving the visitor looking at the textarea.
    if (ruler && typeof ruler.scrollIntoView === "function") {
      var reduce =
        window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      try {
        ruler.scrollIntoView({
          behavior: reduce ? "auto" : "smooth",
          block: "start",
        });
      } catch (e) {
        /* older engines take no options — the result is still on screen */
        ruler.scrollIntoView();
      }
    }
  }

  /* --------------------------------------------------------- activation */

  // Profile is the only thing that produces numbers. Everything else just
  // supplies input, so a visitor can always tell which action caused what.
  function hasInput() {
    return input.value.trim().length > 0;
  }

  function syncButton() {
    run.setAttribute("aria-disabled", hasInput() ? "false" : "true");
  }

  // aria-disabled rather than the disabled attribute: the control stays
  // focusable and announced, so a keyboard user meets it and hears why it
  // is unavailable instead of finding a gap in the tab order.
  run.addEventListener("click", function () {
    if (!hasInput()) {
      if (status) status.textContent = "paste or load something first";
      input.focus();
      return;
    }
    render(input.value);
  });

  // Typing only toggles the button — it never profiles.
  input.addEventListener("input", syncButton);

  /* --------------------------------------------------------- input paths */

  var SAMPLES = {
    json: function () {
      var rows = [];
      for (var i = 0; i < 120; i++) {
        rows.push({
          id: 4800 + i,
          name: "user" + i,
          email: "user" + i + "@example.com",
          role: i % 7 === 0 ? "admin" : "member",
          city: "Madison",
          plan: "pro",
        });
      }
      return JSON.stringify(rows, null, 2);
    },
    logs: function () {
      var out = [];
      for (var i = 0; i < 120; i++) {
        out.push(
          "2026-06-26T07:26:" +
            String(i % 60).padStart(2, "0") +
            ".730Z ERROR [billing] reqId=" +
            (1898747 + i) +
            " token validated latency_ms=" +
            (700 + (i % 300)) +
            " status=500"
        );
      }
      return out.join("\n");
    },
    csv: function () {
      var out = ["id,name,city,plan"];
      for (var i = 0; i < 120; i++) {
        out.push([4800 + i, "user" + i, "madison", "pro"].join(","));
      }
      return out.join("\n");
    },
  };

  Array.prototype.forEach.call(
    document.querySelectorAll("[data-sample]"),
    function (btn) {
      btn.addEventListener("click", function () {
        var make = SAMPLES[btn.getAttribute("data-sample")];
        if (!make) return;
        input.value = make();
        syncButton();
        reset();
        if (status) status.textContent = "sample loaded — press Profile to run it";
        input.focus();
      });
    }
  );

  var clear = document.getElementById("pf-clear");
  if (clear) {
    clear.addEventListener("click", function () {
      input.value = "";
      syncButton();
      reset();
    });
  }

  var file = document.getElementById("pf-file");
  function readFile(f) {
    if (!f) return;
    var r = new FileReader();
    r.onload = function () {
      input.value = String(r.result);
      syncButton();
      reset();
      if (status) status.textContent = (f.name || "file") + " loaded — press Profile to run it";
    };
    r.readAsText(f);
  }

  if (file) {
    file.addEventListener("change", function () {
      readFile(file.files && file.files[0]);
    });
  }

  if (wrap) {
    ["dragenter", "dragover"].forEach(function (ev) {
      wrap.addEventListener(ev, function (e) {
        e.preventDefault();
        wrap.classList.add("drag");
      });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      wrap.addEventListener(ev, function (e) {
        e.preventDefault();
        wrap.classList.remove("drag");
      });
    });
    wrap.addEventListener("drop", function (e) {
      readFile(e.dataTransfer && e.dataTransfer.files[0]);
    });
  }
  // Blank whatever the no-JS markup left behind and start from zero.
  syncButton();
  reset();
})();
