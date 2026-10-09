/* PDF RAG web UI -- form handling, loader, and result rendering. */
(function () {
  "use strict";

  var form = document.getElementById("ask-form");
  var input = document.getElementById("question");
  var askButton = document.getElementById("ask-button");
  var askLabel = askButton.querySelector(".btn__label");
  var charCount = document.getElementById("char-count");

  var loader = document.getElementById("loader");
  var loaderStep = document.getElementById("loader-step");

  var errorBox = document.getElementById("error");
  var errorText = document.getElementById("error-text");

  var result = document.getElementById("result");
  var resultBadge = document.getElementById("result-badge");
  var resultQuestion = document.getElementById("result-question");
  var answerBox = document.getElementById("answer");
  var resultSources = document.getElementById("result-sources");
  var resultScore = document.getElementById("result-score");
  var resultThreshold = document.getElementById("result-threshold");
  var resultElapsed = document.getElementById("result-elapsed");

  var chunksWrap = document.getElementById("chunks-wrap");
  var chunksCount = document.getElementById("chunks-count");
  var chunksList = document.getElementById("chunks");

  var statusBanner = document.getElementById("status-banner");
  var statusText = document.getElementById("status-text");

  var MAX_LENGTH = parseInt(input.getAttribute("maxlength"), 10) || 1000;
  var inFlight = false;
  var stepTimer = null;

  /* --- helpers --------------------------------------------------------- */

  function show(el) { el.hidden = false; }
  function hide(el) { el.hidden = true; }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /* Cohere replies in light markdown. Render a safe subset: the text is
     HTML-escaped first, so only the tags produced below can ever reach the
     DOM. No user or model content is inserted as raw HTML. */
  function renderMarkdown(text) {
    var lines = escapeHtml(text).replace(/\r\n/g, "\n").split("\n");
    var html = "";
    var listType = null;

    function closeList() {
      if (listType) { html += "</" + listType + ">"; listType = null; }
    }

    function openList(type) {
      if (listType !== type) { closeList(); html += "<" + type + ">"; listType = type; }
    }

    function inline(s) {
      return s
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    }

    lines.forEach(function (line) {
      var trimmed = line.trim();

      /* A blank line does not end a list. Cohere separates list items with
         blank lines, and closing here would split one list into several. */
      if (!trimmed) { return; }

      var heading = /^#{1,6}\s+(.*)$/.exec(trimmed);
      if (heading) {
        closeList();
        html += "<h3>" + inline(heading[1]) + "</h3>";
        return;
      }

      var bullet = /^[-*+]\s+(.*)$/.exec(trimmed);
      if (bullet) {
        openList("ul");
        html += "<li>" + inline(bullet[1]) + "</li>";
        return;
      }

      /* Carry the source number through as `value`. Nested bullets between two
         numbered items force a new <ol>, which would otherwise restart at 1 —
         this keeps 1, 2, 3 intact. The capture is \d+ only, so it is safe to
         drop straight into the attribute. */
      var numbered = /^(\d+)[.)]\s+(.*)$/.exec(trimmed);
      if (numbered) {
        openList("ol");
        html += '<li value="' + numbered[1] + '">' + inline(numbered[2]) + "</li>";
        return;
      }

      closeList();
      html += "<p>" + inline(trimmed) + "</p>";
    });

    closeList();
    return html || "<p></p>";
  }

  function setBusy(busy) {
    inFlight = busy;
    askButton.disabled = busy;
    input.readOnly = busy;
    askLabel.textContent = busy ? "Searching\u2026" : "Search document";
    form.setAttribute("aria-busy", busy ? "true" : "false");

    if (busy) {
      hide(errorBox);
      hide(result);
      show(loader);
      startSteps();
    } else {
      hide(loader);
      stopSteps();
    }
  }

  /* Rotating sub-caption so the loader reflects the real pipeline stages. */
  function startSteps() {
    var steps = [
      "Embedding your question",
      "Searching the FAISS index",
      "Checking retrieval relevance",
      "Asking Cohere for an answer",
      "Still working, almost there"
    ];
    var i = 0;
    loaderStep.textContent = steps[0];
    stepTimer = window.setInterval(function () {
      i = Math.min(i + 1, steps.length - 1);
      loaderStep.textContent = steps[i];
    }, 1800);
  }

  function stopSteps() {
    if (stepTimer) { window.clearInterval(stepTimer); stepTimer = null; }
  }

  function showError(message) {
    errorText.textContent = message;
    show(errorBox);
    hide(result);
  }

  /* Distances need the precision; the configured threshold reads better as the
     value the user actually set (1.7, not 1.7000). */
  function fmt(n) { return Number(n).toFixed(4); }
  function fmtPlain(n) { return String(Number(n)); }

  /* --- rendering ------------------------------------------------------- */

  function renderResult(data, elapsedMs) {
    resultQuestion.textContent = data.question;
    answerBox.innerHTML = renderMarkdown(data.answer);

    if (data.answered) {
      resultBadge.textContent = "Match found";
      resultBadge.className = "badge badge--ok";
    } else {
      resultBadge.textContent = "No relevant match";
      resultBadge.className = "badge badge--warn";
    }

    resultSources.textContent =
      data.sources && data.sources.length ? data.sources.join(", ") : "\u2014";
    resultScore.textContent =
      data.scores && data.scores.length ? fmt(data.best_score) : "\u2014";
    resultThreshold.textContent = fmtPlain(data.threshold);
    resultElapsed.textContent = (elapsedMs / 1000).toFixed(1) + " s";

    chunksList.textContent = "";
    if (data.chunks && data.chunks.length) {
      data.chunks.forEach(function (chunk, i) {
        var li = document.createElement("li");

        if (data.scores && data.scores[i] !== undefined) {
          var score = document.createElement("span");
          score.className = "chunk__score";
          score.textContent = "distance " + fmt(data.scores[i]);
          li.appendChild(score);
        }

        var p = document.createElement("p");
        p.className = "chunk__text";
        p.textContent = chunk;
        li.appendChild(p);

        chunksList.appendChild(li);
      });
      chunksCount.textContent = "(" + data.chunks.length + ")";
      show(chunksWrap);
    } else {
      hide(chunksWrap);
    }

    show(result);
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /* --- requests -------------------------------------------------------- */

  function ask(question) {
    var started = performance.now();
    setBusy(true);

    fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question })
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, status: response.status, data: data };
        }).catch(function () {
          throw new Error("Server returned a non-JSON response (HTTP " + response.status + ").");
        });
      })
      .then(function (payload) {
        if (!payload.ok) {
          throw new Error(payload.data.error || "Request failed (HTTP " + payload.status + ").");
        }
        renderResult(payload.data, performance.now() - started);
      })
      .catch(function (err) {
        showError(err.message || "Could not reach the server.");
      })
      .finally(function () {
        setBusy(false);
        input.focus();
      });
  }

  function checkStatus() {
    fetch("/api/status")
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (p) {
        if (p.ok && p.data.ready) {
          statusBanner.className = "banner banner--ok";
          statusText.textContent =
            "Ready \u2014 " + p.data.settings.vectors + " vectors indexed from " +
            p.data.settings.pdf + ".";
        } else {
          statusBanner.className = "banner banner--error";
          statusText.textContent = p.data.error || "The pipeline is not ready.";
          askButton.disabled = true;
        }
      })
      .catch(function () {
        statusBanner.className = "banner banner--error";
        statusText.textContent = "Could not reach the server.";
      });
  }

  /* --- events ---------------------------------------------------------- */

  function updateCharCount() {
    charCount.textContent = input.value.length + " / " + MAX_LENGTH;
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (inFlight) return;

    var question = input.value.trim();
    if (!question) {
      input.setAttribute("aria-invalid", "true");
      showError("Please enter a question first.");
      input.focus();
      return;
    }

    input.removeAttribute("aria-invalid");
    ask(question);
  });

  form.addEventListener("reset", function () {
    hide(result);
    hide(errorBox);
    input.removeAttribute("aria-invalid");
    window.setTimeout(function () { updateCharCount(); input.focus(); }, 0);
  });

  input.addEventListener("input", function () {
    updateCharCount();
    if (input.value.trim()) input.removeAttribute("aria-invalid");
  });

  input.addEventListener("keydown", function (event) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  document.getElementById("examples").addEventListener("click", function (event) {
    var chip = event.target.closest(".chip");
    if (!chip || inFlight) return;
    input.value = chip.textContent.trim();
    updateCharCount();
    form.requestSubmit();
  });

  updateCharCount();
  checkStatus();
})();
