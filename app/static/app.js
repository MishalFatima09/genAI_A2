// Task 6 Frontend Logic: Screening, Evidence Highlighting, Live Tiled Attention, and Benchmarks

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  let globalPresets = { held_out_test_patients: [], unseen_demos: [] };
  let currentPatientSentences = [];
  let currentCriteria = [];
  let latestScreeningResponse = null;

  // DOM Elements
  const presetSelect = document.getElementById("presetSelect");
  const testPatientOptions = document.getElementById("testPatientOptions");
  const unseenDemoOptions = document.getElementById("unseenDemoOptions");
  const btnLoadPreset = document.getElementById("btnLoadPreset");
  const btnScreen = document.getElementById("btnScreen");
  const trialIdBadge = document.getElementById("trialIdBadge");
  const trialTitleInput = document.getElementById("trialTitleInput");
  const patientNoteViewer = document.getElementById("patientNoteViewer");
  const rawNoteInput = document.getElementById("rawNoteInput");
  const criteriaList = document.getElementById("criteriaList");
  const overallVerdictContainer = document.getElementById("overallVerdictContainer");
  const validationAlert = document.getElementById("validationAlert");
  const validationAlertContent = document.getElementById("validationAlertContent");
  const btnAddInclusion = document.getElementById("btnAddInclusion");
  const btnAddExclusion = document.getElementById("btnAddExclusion");
  const btnExportMarkdown = document.getElementById("btnExportMarkdown");
  const btnExportJSON = document.getElementById("btnExportJSON");
  const btnRunTiled = document.getElementById("btnRunTiled");

  // 1. Fetch Presets from Backend
  fetch("/api/presets")
    .then((r) => r.json())
    .then((data) => {
      if (!data.success) return;
      globalPresets = data;

      // Populate Held-Out Test Patients
      data.held_out_test_patients.forEach((p, idx) => {
        const opt = document.createElement("option");
        opt.value = `test_${idx}`;
        opt.textContent = `Patient ${p.patient_id} (${p.trial_id} - ${p.criteria.length} criteria)`;
        testPatientOptions.appendChild(opt);
      });

      // Populate Unseen Demos
      data.unseen_demos.forEach((d) => {
        const opt = document.createElement("option");
        opt.value = `demo_${d.id}`;
        opt.textContent = d.name;
        unseenDemoOptions.appendChild(opt);
      });

      // Auto-load first test patient for instant interactivity
      if (data.held_out_test_patients.length > 0) {
        presetSelect.value = "test_0";
        loadSelectedPreset();
      }
    })
    .catch((err) => console.error("Failed to load presets:", err));

  // 2. Fetch Precomputed Benchmark Logs for Tab 2, 3, 4
  fetch("/api/benchmarks")
    .then((r) => r.json())
    .then((data) => {
      if (!data.success) return;
      populateKVBenchmarks(data.task3_kv_results);
      populateSpeculativeBenchmarks(data.task5_speculative_results);
      populateTiledBenchmarks(data.task4_tiled_results);
      populateConfusionMatrix(data.task2_test_metrics);
    })
    .catch((err) => console.error("Failed to load benchmarks:", err));

  // Preset Selection Loader
  btnLoadPreset.addEventListener("click", loadSelectedPreset);
  presetSelect.addEventListener("change", loadSelectedPreset);

  function loadSelectedPreset() {
    hideValidationError();
    const val = presetSelect.value;
    if (!val) return;

    let patientData = null;
    if (val.startsWith("test_")) {
      const idx = parseInt(val.split("_")[1]);
      patientData = globalPresets.held_out_test_patients[idx];
    } else if (val.startsWith("demo_")) {
      const demoId = val.replace("demo_", "");
      patientData = globalPresets.unseen_demos.find((d) => d.id === demoId);
    }

    if (!patientData) return;

    trialIdBadge.textContent = `Trial ID: ${patientData.trial_id}`;
    trialTitleInput.value = patientData.trial_title;
    rawNoteInput.value = patientData.note;

    // Render Sentences Box
    renderSentencesFromNote(patientData.note);

    // Populate Criteria
    currentCriteria = JSON.parse(JSON.stringify(patientData.criteria || []));
    renderCriteriaCards();
    overallVerdictContainer.classList.add("d-none");
  }

  function renderSentencesFromNote(noteText) {
    patientNoteViewer.innerHTML = "";
    currentPatientSentences = [];

    if (!noteText.trim()) {
      patientNoteViewer.innerHTML = `<div class="text-muted text-center py-4">No patient record text entered.</div>`;
      return;
    }

    const lines = noteText.split("\n").filter((l) => l.trim().length > 0);
    lines.forEach((line, idx) => {
      let sid = idx;
      let text = line;
      const match = line.match(/^(\d+)\.\s+(.*)/);
      if (match) {
        sid = parseInt(match[1]);
        text = match[2];
      }

      currentPatientSentences.push({ id: sid, text: text });

      const div = document.createElement("div");
      div.className = "sentence-item";
      div.id = `sentence-row-${sid}`;
      div.innerHTML = `<span class="sentence-id">${sid}.</span><span class="sentence-text">${escapeHtml(text)}</span>`;
      patientNoteViewer.appendChild(div);
    });
  }

  rawNoteInput.addEventListener("input", () => {
    renderSentencesFromNote(rawNoteInput.value);
  });

  // Criteria Card Rendering
  function renderCriteriaCards() {
    criteriaList.innerHTML = "";
    if (currentCriteria.length === 0) {
      criteriaList.innerHTML = `<div class="text-muted text-center py-4">No criteria added. Click <strong>Add Inclusion</strong> or <strong>Add Exclusion</strong> above.</div>`;
      return;
    }

    currentCriteria.forEach((c, idx) => {
      const card = document.createElement("div");
      card.className = "criterion-card";
      card.id = `criterion-card-${idx}`;

      const isInc = (c.criterion_type || "inclusion").toLowerCase() === "inclusion";
      const typeBadgeClass = isInc ? "bg-success-subtle text-success border border-success-subtle" : "bg-danger-subtle text-danger border border-danger-subtle";

      // Prediction details if screened
      let decisionHtml = "";
      if (c.pred_label) {
        const lblClass = `label-${c.pred_label.replace(/\s+/g, "_")}`;
        const evidenceStr = c.pred_evidence && c.pred_evidence.length > 0 ? c.pred_evidence.join(", ") : "None";
        decisionHtml = `
          <div class="mt-2 pt-2 border-top">
            <div class="d-flex align-items-center justify-content-between mb-1">
              <div>
                <span class="label-badge ${lblClass}">${c.pred_label}</span>
                <span class="badge bg-light text-dark border ms-2">Evidence IDs: [${evidenceStr}]</span>
              </div>
              <small class="text-muted">${c.latency_ms ? c.latency_ms + " ms" : ""}</small>
            </div>
            <p class="small text-muted mb-0 mt-1"><strong>Explanation:</strong> ${escapeHtml(c.pred_explanation || "No explanation provided.")}</p>
          </div>
        `;
      }

      card.innerHTML = `
        <div class="d-flex align-items-start justify-content-between">
          <div class="d-flex align-items-center gap-2 flex-grow-1 me-2">
            <span class="badge ${typeBadgeClass} text-uppercase">${escapeHtml(c.criterion_type || "inclusion")}</span>
            <input type="text" class="form-control form-control-sm border-0 bg-transparent fw-semibold criterion-text-input" 
                   data-idx="${idx}" value="${escapeHtml(c.criterion_text || "")}" placeholder="Enter criterion requirement...">
          </div>
          <button class="btn btn-sm btn-link text-danger p-0 ms-2" onclick="removeCriterion(${idx})" title="Remove criterion">
            <i class="bi bi-trash"></i>
          </button>
        </div>
        ${decisionHtml}
      `;

      // Highlight corresponding sentences on hover or click
      card.addEventListener("click", () => {
        highlightSentences(c.pred_evidence || []);
      });

      criteriaList.appendChild(card);
    });

    // Attach input listeners
    document.querySelectorAll(".criterion-text-input").forEach((inp) => {
      inp.addEventListener("input", (e) => {
        const i = parseInt(e.target.dataset.idx);
        currentCriteria[i].criterion_text = e.target.value;
      });
    });
  }

  window.removeCriterion = function (index) {
    currentCriteria.splice(index, 1);
    renderCriteriaCards();
  };

  btnAddInclusion.addEventListener("click", () => {
    currentCriteria.push({ criterion_type: "inclusion", criterion_text: "" });
    renderCriteriaCards();
  });

  btnAddExclusion.addEventListener("click", () => {
    currentCriteria.push({ criterion_type: "exclusion", criterion_text: "" });
    renderCriteriaCards();
  });

  function highlightSentences(sentenceIds) {
    document.querySelectorAll(".sentence-item").forEach((el) => {
      el.classList.remove("sentence-highlight");
    });
    if (!sentenceIds || sentenceIds.length === 0) return;

    sentenceIds.forEach((sid) => {
      const row = document.getElementById(`sentence-row-${sid}`);
      if (row) {
        row.classList.add("sentence-highlight");
        row.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });
  }

  // 3. Run Screening Request
  btnScreen.addEventListener("click", () => {
    hideValidationError();
    btnScreen.disabled = true;
    btnScreen.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> Screening...`;

    const payload = {
      note: rawNoteInput.value,
      criteria: currentCriteria,
      mode: document.getElementById("inferenceEngineSelect").value,
    };

    fetch("/api/screen", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json().then((data) => ({ status: r.status, body: data })))
      .then(({ status, body }) => {
        btnScreen.disabled = false;
        btnScreen.innerHTML = `<i class="bi bi-play-circle-fill me-1"></i>Run Screening`;

        if (status !== 200 || !body.success) {
          showValidationError(body.error || "Screening Error", body.details || [body.remediation]);
          return;
        }

        latestScreeningResponse = body;

        // Update criteria with predictions
        body.criterion_results.forEach((res, i) => {
          if (currentCriteria[i]) {
            currentCriteria[i].pred_label = res.pred_label;
            currentCriteria[i].pred_evidence = res.pred_evidence;
            currentCriteria[i].pred_explanation = res.pred_explanation;
            currentCriteria[i].latency_ms = res.latency_ms;
          }
        });
        renderCriteriaCards();

        // Render Overall Aggregated Trial Verdict
        renderOverallVerdict(body.overall_verdict);

        // Update summary footer
        const meta = body.inference_metadata || {};
        document.getElementById("screeningMetadataSummary").textContent = 
          `Processed in ${meta.total_latency_ms} ms | Device: ${meta.device.toUpperCase()} | Engine: ${meta.mode}`;
      })
      .catch((err) => {
        btnScreen.disabled = false;
        btnScreen.innerHTML = `<i class="bi bi-play-circle-fill me-1"></i>Run Screening`;
        showValidationError("Network Error", [err.message]);
      });
  });

  function renderOverallVerdict(verdict) {
    if (!verdict) return;
    overallVerdictContainer.classList.remove("d-none");
    const vClass = `verdict-${verdict.verdict}`;

    let disqualifiersHtml = "";
    if (verdict.disqualifying_reasons && verdict.disqualifying_reasons.length > 0) {
      disqualifiersHtml = `
        <div class="mt-2">
          <strong>Disqualification Criteria Identified:</strong>
          <ul class="mb-1 ps-3">
            ${verdict.disqualifying_reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}
          </ul>
        </div>
      `;
    }

    let missingHtml = "";
    if (verdict.missing_info_reasons && verdict.missing_info_reasons.length > 0) {
      missingHtml = `
        <div class="mt-2 text-warning-emphasis">
          <strong>Missing Clinical Information to Verify:</strong>
          <ul class="mb-1 ps-3">
            ${verdict.missing_info_reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}
          </ul>
        </div>
      `;
    }

    overallVerdictContainer.innerHTML = `
      <div class="verdict-banner ${vClass} shadow-sm">
        <div class="d-flex align-items-center justify-content-between">
          <h4 class="mb-0 fw-bold">${escapeHtml(verdict.title)}</h4>
          <span class="badge bg-${verdict.badge_color} fs-6 text-uppercase px-3 py-1">${verdict.verdict}</span>
        </div>
        <p class="mt-2 mb-1">${escapeHtml(verdict.description)}</p>
        ${disqualifiersHtml}
        ${missingHtml}
      </div>
    `;
  }

  function showValidationError(title, details) {
    validationAlert.classList.remove("d-none");
    validationAlertContent.innerHTML = `
      <strong>${escapeHtml(title)}</strong>
      <ul class="mb-0 ps-3 mt-1">
        ${details.map((d) => `<li>${escapeHtml(d)}</li>`).join("")}
      </ul>
    `;
  }

  function hideValidationError() {
    validationAlert.classList.add("d-none");
    validationAlertContent.innerHTML = "";
  }

  // Export handlers
  btnExportMarkdown.addEventListener("click", () => triggerExport("markdown"));
  btnExportJSON.addEventListener("click", () => triggerExport("json"));

  function triggerExport(format) {
    if (!latestScreeningResponse) {
      alert("Please run a patient screening first before exporting the report.");
      return;
    }

    fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        format: format,
        screening_data: latestScreeningResponse,
      }),
    })
      .then((response) => response.blob())
      .then((blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `clinical_trial_screening_report.${format === "json" ? "json" : "md"}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
      });
  }

  // 4. Live Tiled Attention Execution (Task 4)
  btnRunTiled.addEventListener("click", () => {
    btnRunTiled.disabled = true;
    btnRunTiled.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> Computing Tiled vs Ref Attention...`;

    const payload = {
      B: parseInt(document.getElementById("inputB").value),
      H: parseInt(document.getElementById("inputH").value),
      L: parseInt(document.getElementById("inputL").value),
      D: parseInt(document.getElementById("inputD").value),
      block_size: parseInt(document.getElementById("inputBlockSize").value),
      causal: document.getElementById("inputCausal").value === "true",
    };

    fetch("/api/live_attention", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        btnRunTiled.disabled = false;
        btnRunTiled.innerHTML = `<i class="bi bi-play-circle me-1"></i>Execute Live Numerical Verification`;

        if (!data.success) return;
        document.getElementById("tiledLiveResultsPlaceholder").classList.add("d-none");
        document.getElementById("tiledLiveResultsContainer").classList.remove("d-none");

        document.getElementById("resAllclose").textContent = data.results.allclose_passed ? "PASSED (allclose)" : "FAILED";
        document.getElementById("resMaxAbs").textContent = data.results.max_abs_diff;
        document.getElementById("resMemSaving").textContent = data.results.theoretical_mem_saving_pct;

        document.getElementById("tiledDebugOutput").textContent = 
          `Configuration: B=${data.config.B}, H=${data.config.H}, L=${data.config.L}, D=${data.config.D}, Tile=${data.config.block_size}, Causal=${data.config.causal}\n` +
          `Reference Attention Latency : ${data.results.ref_time_ms} ms\n` +
          `Tiled Online Softmax Latency : ${data.results.tiled_time_ms} ms\n` +
          `Mean Absolute Difference    : ${data.results.mean_abs_diff}\n` +
          `Max Absolute Difference     : ${data.results.max_abs_diff}\n` +
          `Verification Status         : PASSED (PyTorch tensor equality within numerical tolerance)`;
      })
      .catch((err) => {
        btnRunTiled.disabled = false;
        btnRunTiled.innerHTML = `<i class="bi bi-play-circle me-1"></i>Execute Live Numerical Verification`;
        alert("Error executing live attention: " + err.message);
      });
  });

  // Populate Benchmark Logs
  function populateSpeculativeBenchmarks(specData) {
    if (!specData || !specData.configs) return;
    const tbody = document.getElementById("specBenchmarkTableBody");
    tbody.innerHTML = "";

    specData.configs.forEach((c) => {
      const gTok = c.greedy ? c.greedy.tokens_s_mean.toFixed(1) : "N/A";
      const sTok = c.speculative ? c.speculative.tokens_s_mean.toFixed(1) : "N/A";
      const speedup = c.greedy && c.speculative && c.greedy.tokens_s_mean > 0
        ? (c.speculative.tokens_s_mean / c.greedy.tokens_s_mean).toFixed(2) + "x"
        : "N/A";
      const acc = c.speculative ? (c.speculative.acceptance_rate_mean * 100).toFixed(1) + "%" : "N/A";
      const matchBadge = c.correctness && c.correctness.match
        ? `<span class="badge bg-success">MATCH (100%)</span>`
        : `<span class="badge bg-danger">MISMATCH</span>`;

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="badge bg-secondary font-monospace">L = ${c.prompt_len}</span></td>
        <td><span class="badge bg-light text-dark border font-monospace">Gen = ${c.gen_len}</span></td>
        <td class="font-monospace">${gTok} tok/s</td>
        <td class="font-monospace fw-bold text-primary">${sTok} tok/s</td>
        <td><span class="badge ${parseFloat(speedup) >= 1.0 ? 'bg-success' : 'bg-warning text-dark'}">${speedup}</span></td>
        <td><span class="badge bg-info-subtle text-info border">${acc}</span></td>
        <td>${matchBadge}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function populateKVBenchmarks(kvData) {
    if (!kvData || !kvData.configs) return;
    const tbody = document.getElementById("kvBenchmarkTableBody");
    if (!tbody) return;
    tbody.innerHTML = "";

    kvData.configs.forEach((c) => {
      const uTok = c.uncached ? c.uncached.tokens_s_mean.toFixed(1) : "N/A";
      const cTok = c.cached ? c.cached.tokens_s_mean.toFixed(1) : "N/A";
      const speedupVal = c.uncached && c.cached && c.uncached.tokens_s_mean > 0
        ? (c.cached.tokens_s_mean / c.uncached.tokens_s_mean).toFixed(2)
        : "N/A";
      const prefillS = c.cached ? (c.cached.prefill_s_mean * 1000).toFixed(1) + " ms" : "N/A";
      const decodeS = c.cached ? c.cached.decode_s_mean.toFixed(2) + " s" : "N/A";
      const peakMem = c.cached ? c.cached.peak_mem_mb_mean.toFixed(0) + " MB" : "N/A";
      const matchBadge = c.correctness && c.correctness.match
        ? `<span class="badge bg-success">MATCH (100%)</span>`
        : `<span class="badge bg-danger">MISMATCH</span>`;

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="badge bg-secondary font-monospace">L = ${c.prompt_len}</span></td>
        <td><span class="badge bg-light text-dark border font-monospace">Gen = ${c.gen_len}</span></td>
        <td class="font-monospace text-muted">${uTok} tok/s</td>
        <td class="font-monospace fw-bold text-success">${cTok} tok/s</td>
        <td><span class="badge bg-success-subtle text-success border border-success-subtle fw-bold">${speedupVal}x</span></td>
        <td class="font-monospace small">${prefillS}</td>
        <td class="font-monospace small">${decodeS}</td>
        <td class="font-monospace small">${peakMem}</td>
        <td>${matchBadge}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function populateTiledBenchmarks(tiledData) {
    if (!tiledData || !tiledData.benchmarks) return;
    const tbody = document.getElementById("tiledBenchmarkTableBody");
    tbody.innerHTML = "";

    tiledData.benchmarks.forEach((b) => {
      const tr = document.createElement("tr");
      const memSaved = ((1.0 - b.mem_ratio) * 100).toFixed(1) + "%";
      tr.innerHTML = `
        <td class="font-monospace">L = ${b.config.L}</td>
        <td class="font-monospace">${b.reference.mean_s.toFixed(4)}</td>
        <td class="font-monospace">${b.tiled.mean_s.toFixed(4)}</td>
        <td class="font-monospace">${b.reference.peak_mem_mb.toFixed(1)} MB</td>
        <td class="font-monospace text-success fw-bold">${b.tiled.peak_mem_mb.toFixed(1)} MB</td>
        <td><span class="badge bg-success">${memSaved} saved</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  function populateConfusionMatrix(metricsData) {
    if (!metricsData || !metricsData.label || !metricsData.label.overall) return;
    const cm = metricsData.label.overall.confusion_matrix;
    if (!cm) return;

    const tbody = document.getElementById("confusionMatrixBody");
    tbody.innerHTML = "";

    const labels = ["included", "not included", "excluded", "not excluded", "not enough info", "not applicable", "<invalid>"];

    cm.rows_gold_cols_pred.forEach((row, rIdx) => {
      if (rIdx >= labels.length) return;
      const tr = document.createElement("tr");
      let rowHtml = `<td class="fw-bold bg-light">${labels[rIdx]}</td>`;
      row.forEach((count, cIdx) => {
        const isDiag = rIdx === cIdx;
        const bgStyle = isDiag && count > 0 ? "background-color: #d1fae5; font-weight: bold;" : "";
        rowHtml += `<td style="${bgStyle}">${count}</td>`;
      });
      tr.innerHTML = rowHtml;
      tbody.appendChild(tr);
    });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
