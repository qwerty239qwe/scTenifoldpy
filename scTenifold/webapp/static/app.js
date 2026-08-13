"use strict";

const state = {
  workflow: "net",
  datasets: { x: null, y: null },
  allGenes: [],
  koGenes: new Set(),
  // Bumped whenever the form no longer describes an in-flight job (i.e. on a
  // workflow switch), so that job stops writing to the results UI.
  runGeneration: 0,
};

const $ = (id) => document.getElementById(id);

const DATASET_SECTION_HINTS = {
  net: "scTenifoldNet compares two samples — e.g. control vs. treatment. Provide a Dataset X (first condition) and a Dataset Y (second condition) below.",
  knk: "scTenifoldKnk only needs a single sample: it simulates knocking a gene out of Dataset X and compares the gene network before/after.",
  grn: "Just infers the gene regulatory network from a single sample (Dataset X) — no second condition, no knockout.",
};

const DATASET_X_ROLES = {
  net: "— the first condition",
  knk: "— the sample to knock a gene out of",
  grn: "— the sample to build a network from",
};

function setWorkflow(workflow) {
  state.workflow = workflow;
  const isNet = workflow === "net";
  const isKnk = workflow === "knk";
  const isGrn = workflow === "grn";

  $("dataset-y-slot").hidden = !isNet;
  $("y-label-field").hidden = !isNet;
  $("x-label-field").hidden = !isNet;
  $("ko-genes-group").hidden = !isKnk;
  $("ko-method-field").hidden = !isKnk;
  $("strict-lambda-field").hidden = !isKnk;
  // A single network has nothing to resample across, so parallel backend
  // choice doesn't apply.
  $("backend-field").hidden = isGrn;
  $("n-jobs-field").hidden = isGrn;

  $("dataset-section-hint").textContent = DATASET_SECTION_HINTS[workflow];
  $("dataset-x-role").textContent = DATASET_X_ROLES[workflow];

  if (!isKnk) {
    state.koGenes.clear();
    renderKoGeneChips();
  }
  state.runGeneration += 1;
  resetResults();
  updateRunReadiness();
}

function resetResults() {
  $("results-section").classList.add("hidden");
  $("results-error").classList.add("hidden");
  $("results-error").textContent = "";
  $("results-table-wrap").innerHTML = "";
  $("download-csv").classList.add("hidden");
  $("progress-wrap").classList.add("hidden");
  $("progress-fill").style.width = "10%";
  $("progress-fill").classList.remove("error");
  $("status-bar").textContent = "";
}

async function apiFetch(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try {
    body = await res.json();
  } catch (err) {
    // no JSON body (e.g. CSV download)
  }
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : res.statusText;
    throw new Error(detail);
  }
  return body;
}

function renderDatasetInfo(slot, info, errorMessage) {
  const infoEl = $(`dataset-${slot}-info`);
  const errorEl = $(`dataset-${slot}-error`);
  if (errorMessage) {
    errorEl.textContent = errorMessage;
    errorEl.classList.remove("hidden");
    infoEl.classList.add("hidden");
    return;
  }
  errorEl.classList.add("hidden");
  infoEl.textContent = `${info.name}: ${info.n_genes} genes x ${info.n_cells} cells`;
  infoEl.classList.remove("hidden");
}

function populateGeneList(geneNames) {
  state.allGenes = geneNames;
  renderKoGeneOptions();
}

function renderKoGeneOptions() {
  const select = $("ko-gene-select");
  const filter = $("ko-gene-filter").value.trim().toLowerCase();
  const genes = filter
    ? state.allGenes.filter((g) => g.toLowerCase().includes(filter))
    : state.allGenes;

  select.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const gene of genes) {
    const option = document.createElement("option");
    option.value = gene;
    option.textContent = gene;
    option.selected = state.koGenes.has(gene);
    fragment.appendChild(option);
  }
  select.appendChild(fragment);
}

function onKoGeneSelectChange() {
  // Only options currently rendered (i.e. matching the active filter) can
  // toggle membership here; genes hidden by the filter keep their prior
  // selection state untouched (see renderKoGeneOptions).
  for (const option of $("ko-gene-select").options) {
    if (option.selected) {
      state.koGenes.add(option.value);
    } else {
      state.koGenes.delete(option.value);
    }
  }
  renderKoGeneChips();
}

function setUploadGridVisible(visible) {
  $("dataset-grid").hidden = !visible;
  $("toggle-upload").setAttribute("aria-expanded", String(visible));
}

function toggleUploadGrid() {
  setUploadGridVisible($("dataset-grid").hidden);
}

async function uploadDataset(slot, file) {
  const formData = new FormData();
  formData.append("file", file);
  try {
    const info = await apiFetch("/api/datasets", { method: "POST", body: formData });
    state.datasets[slot] = info;
    renderDatasetInfo(slot, info);
    if (slot === "x") populateGeneList(info.gene_names);
  } catch (err) {
    state.datasets[slot] = null;
    renderDatasetInfo(slot, null, err.message);
  }
  updateRunReadiness();
}

const EXAMPLE_HINTS = {
  "/api/datasets/example": "",
  "/api/datasets/pbmc3k":
    "Real 10x PBMC3k data (Seurat/Scanpy tutorial dataset), QC-filtered and downsampled for speed. " +
    "For the 'net' workflow it's split into two random halves — a demo of the pipeline, not two real conditions. " +
    "First load downloads ~7 MB and may take a few seconds.",
};

async function useExampleDataset(event) {
  const source = event.currentTarget.dataset.source;
  $("example-hint").textContent = EXAMPLE_HINTS[source] || "";

  // An example replaces any manual upload; collapse the upload boxes and clear
  // both slots (state and their leftover info/error text) so nothing stale
  // lingers — a workflow without a Dataset Y must not keep an old Y active,
  // and a failed load must not leave the previous datasets runnable.
  setUploadGridVisible(false);
  for (const slot of ["x", "y"]) {
    state.datasets[slot] = null;
    $(`dataset-${slot}-error`).classList.add("hidden");
    $(`dataset-${slot}-info`).classList.add("hidden");
    $(`file-input-${slot}`).value = "";
  }
  updateRunReadiness();

  try {
    const [xInfo, yInfo] = await apiFetch(source);
    state.datasets.x = xInfo;
    renderDatasetInfo("x", xInfo);
    populateGeneList(xInfo.gene_names);
    if (state.workflow === "net") {
      state.datasets.y = yInfo;
      renderDatasetInfo("y", yInfo);
    }
  } catch (err) {
    setUploadGridVisible(true);
    renderDatasetInfo("x", null, err.message);
  }
  updateRunReadiness();
}

function updateRunReadiness() {
  const needsY = state.workflow === "net";
  const ready = state.datasets.x !== null && (!needsY || state.datasets.y !== null);
  $("run-section").classList.toggle("hidden", !ready);
}

function renderKoGeneChips() {
  const wrap = $("ko-gene-chips");
  wrap.innerHTML = "";
  for (const gene of state.koGenes) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = gene + " ";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      state.koGenes.delete(gene);
      for (const option of $("ko-gene-select").options) {
        if (option.value === gene) option.selected = false;
      }
      renderKoGeneChips();
    });
    chip.appendChild(remove);
    wrap.appendChild(chip);
  }
}

function buildJobPayload() {
  const payload = {
    workflow: state.workflow,
    dataset_id: state.datasets.x.dataset_id,
    min_lib_size: Number($("min-lib-size").value),
    min_percent: Number($("min-percent").value),
    backend: $("backend").value,
    n_jobs: Number($("n-jobs").value),
    random_state: Number($("random-state").value),
  };
  if (state.workflow === "net") {
    payload.dataset_id_y = state.datasets.y.dataset_id;
    payload.x_label = $("x-label").value || "X";
    payload.y_label = $("y-label").value || "Y";
  } else if (state.workflow === "knk") {
    payload.ko_genes = Array.from(state.koGenes);
    payload.ko_method = $("ko-method").value;
    payload.strict_lambda = Number($("strict-lambda").value);
  }
  // 'grn' needs nothing beyond the base fields (dataset_id, QC, random_state).
  return payload;
}

const STAGE_PROGRESS = { queued: 10, running: 50, done: 100, error: 100 };

function setProgress(status, stage) {
  $("progress-wrap").classList.remove("hidden");
  $("progress-fill").style.width = `${STAGE_PROGRESS[status] ?? 10}%`;
  $("progress-fill").classList.toggle("error", status === "error");
  $("status-bar").textContent = stage ? `${status} — ${stage}` : status;
}

const RESULTS_ROW_LIMIT = 25;

const GENE_COLUMNS = [
  ["gene", "Gene"],
  ["distance", "Distance"],
  ["z", "Z"],
  ["fc", "FC"],
  ["p_value", "p-value"],
  ["adjusted_p_value", "adj. p-value"],
];

const EDGE_COLUMNS = [
  ["source", "Source"],
  ["target", "Target"],
  ["weight", "Weight"],
];

// `workflow` is the one the job was submitted with, not the one the form
// currently shows — the two can differ if the user switches mid-run.
function renderResultsTable(rows, workflow) {
  const wrap = $("results-table-wrap");
  wrap.innerHTML = "";
  const isGrn = workflow === "grn";
  const noun = isGrn ? "edge" : "gene";
  if (rows.length === 0) {
    wrap.textContent = `No ${noun}s in result.`;
    return;
  }
  const columns = isGrn ? EDGE_COLUMNS : GENE_COLUMNS;
  const sortedBy = isGrn ? "|edge weight|" : "p-value";

  const shown = rows.slice(0, RESULTS_ROW_LIMIT);
  const caption = document.createElement("p");
  caption.className = "hint";
  caption.textContent =
    rows.length > shown.length
      ? `Showing top ${shown.length} of ${rows.length} ${noun}s (sorted by ${sortedBy}). Download the CSV for the full table.`
      : `${rows.length} ${noun}${rows.length === 1 ? "" : "s"} (sorted by ${sortedBy}).`;
  wrap.appendChild(caption);

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const [, label] of columns) {
    const th = document.createElement("th");
    th.textContent = label;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of shown) {
    const tr = document.createElement("tr");
    for (const [key] of columns) {
      const td = document.createElement("td");
      const value = row[key];
      td.textContent = typeof value === "number" ? value.toPrecision(4) : value;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  wrap.appendChild(table);
}

// Returns null once `isCurrent()` goes false — the job was abandoned and its
// progress must no longer be shown.
async function pollJob(jobId, isCurrent) {
  while (isCurrent()) {
    const status = await apiFetch(`/api/jobs/${jobId}`);
    if (!isCurrent()) break;
    setProgress(status.status, status.stage);
    if (status.status === "done") return status;
    if (status.status === "error") throw new Error(status.error || "job failed");
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  return null;
}

function showValidationError(message) {
  $("results-section").classList.remove("hidden");
  $("results-error").textContent = message;
  $("results-error").classList.remove("hidden");
}

async function runJob(event) {
  event.preventDefault();
  if (state.workflow === "knk" && state.koGenes.size === 0) {
    showValidationError("Select at least one gene to knock out.");
    return;
  }
  // The number input can't express "-1 or >= 1" via min alone, and an empty
  // field reads back as 0 — both of which the API rejects with a 422.
  const nJobs = Number($("n-jobs").value);
  if (!Number.isInteger(nJobs) || nJobs === 0 || nJobs < -1) {
    showValidationError("# jobs must be -1 (all cores) or a positive whole number.");
    return;
  }

  $("run-button").disabled = true;
  $("results-section").classList.remove("hidden");
  $("results-error").classList.add("hidden");
  $("results-table-wrap").innerHTML = "";
  $("download-csv").classList.add("hidden");
  setProgress("queued", "");

  // Switching workflows mid-run bumps the generation; from then on this job's
  // results and errors belong to a form the user has moved on from, so they
  // are dropped rather than rendered against the new workflow's layout.
  const generation = state.runGeneration;
  const isCurrent = () => state.runGeneration === generation;

  try {
    const payload = buildJobPayload();
    const { job_id: jobId } = await apiFetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (await pollJob(jobId, isCurrent)) {
      const result = await apiFetch(`/api/jobs/${jobId}/result`);
      if (!isCurrent()) return;
      renderResultsTable(result.rows, payload.workflow);
      const downloadLink = $("download-csv");
      downloadLink.href = `/api/jobs/${jobId}/result.csv`;
      downloadLink.classList.remove("hidden");
    }
  } catch (err) {
    if (!isCurrent()) return;
    $("results-error").textContent = err.message;
    $("results-error").classList.remove("hidden");
  } finally {
    $("run-button").disabled = false;
  }
}

function init() {
  for (const radio of document.querySelectorAll('input[name="workflow"]')) {
    radio.addEventListener("change", (e) => setWorkflow(e.target.value));
  }
  $("use-example").addEventListener("click", useExampleDataset);
  $("use-pbmc3k").addEventListener("click", useExampleDataset);
  $("toggle-upload").addEventListener("click", toggleUploadGrid);
  $("file-input-x").addEventListener("change", (e) => {
    if (e.target.files[0]) uploadDataset("x", e.target.files[0]);
  });
  $("file-input-y").addEventListener("change", (e) => {
    if (e.target.files[0]) uploadDataset("y", e.target.files[0]);
  });
  $("ko-gene-filter").addEventListener("input", renderKoGeneOptions);
  $("ko-gene-select").addEventListener("change", onKoGeneSelectChange);
  $("run-form").addEventListener("submit", runJob);

  setWorkflow("net");
}

init();
