"use strict";

const state = {
  workflow: "net",
  datasets: { x: null, y: null },
  koGenes: [],
};

const $ = (id) => document.getElementById(id);

function setWorkflow(workflow) {
  state.workflow = workflow;
  const isNet = workflow === "net";

  $("dataset-y-slot").hidden = !isNet;
  $("y-label-field").hidden = !isNet;
  $("x-label-field").hidden = !isNet;
  $("ko-genes-group").hidden = isNet;
  $("ko-method-field").hidden = isNet;
  $("strict-lambda-field").hidden = isNet;

  if (isNet) {
    state.koGenes = [];
    renderKoGeneChips();
  }
  updateRunReadiness();
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
  const datalist = $("gene-list");
  datalist.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const gene of geneNames) {
    const option = document.createElement("option");
    option.value = gene;
    fragment.appendChild(option);
  }
  datalist.appendChild(fragment);
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

async function useExampleDataset() {
  try {
    const [xInfo, yInfo] = await apiFetch("/api/datasets/example");
    state.datasets.x = xInfo;
    renderDatasetInfo("x", xInfo);
    populateGeneList(xInfo.gene_names);
    if (state.workflow === "net") {
      state.datasets.y = yInfo;
      renderDatasetInfo("y", yInfo);
    }
  } catch (err) {
    renderDatasetInfo("x", null, err.message);
  }
  updateRunReadiness();
}

function updateRunReadiness() {
  const ready =
    state.datasets.x !== null && (state.workflow === "knk" || state.datasets.y !== null);
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
      state.koGenes = state.koGenes.filter((g) => g !== gene);
      renderKoGeneChips();
    });
    chip.appendChild(remove);
    wrap.appendChild(chip);
  }
}

function addKoGeneFromInput() {
  const input = $("ko-gene-input");
  const gene = input.value.trim();
  if (gene && !state.koGenes.includes(gene)) {
    state.koGenes.push(gene);
    renderKoGeneChips();
  }
  input.value = "";
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
  } else {
    payload.ko_genes = state.koGenes;
    payload.ko_method = $("ko-method").value;
    payload.strict_lambda = Number($("strict-lambda").value);
  }
  return payload;
}

const STAGE_PROGRESS = { queued: 10, running: 50, done: 100, error: 100 };

function setProgress(status, stage) {
  $("progress-wrap").classList.remove("hidden");
  $("progress-fill").style.width = `${STAGE_PROGRESS[status] ?? 10}%`;
  $("progress-fill").classList.toggle("error", status === "error");
  $("status-bar").textContent = stage ? `${status} — ${stage}` : status;
}

function renderResultsTable(rows) {
  const wrap = $("results-table-wrap");
  if (rows.length === 0) {
    wrap.textContent = "No genes in result.";
    return;
  }
  const columns = [
    ["gene", "Gene"],
    ["distance", "Distance"],
    ["z", "Z"],
    ["fc", "FC"],
    ["p_value", "p-value"],
    ["adjusted_p_value", "adj. p-value"],
  ];
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
  for (const row of rows) {
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

  wrap.innerHTML = "";
  wrap.appendChild(table);
}

async function pollJob(jobId) {
  while (true) {
    const status = await apiFetch(`/api/jobs/${jobId}`);
    setProgress(status.status, status.stage);
    if (status.status === "done") return status;
    if (status.status === "error") throw new Error(status.error || "job failed");
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

async function runJob(event) {
  event.preventDefault();
  $("run-button").disabled = true;
  $("results-section").classList.remove("hidden");
  $("results-error").classList.add("hidden");
  $("results-table-wrap").innerHTML = "";
  $("download-csv").classList.add("hidden");
  setProgress("queued", "");

  try {
    const payload = buildJobPayload();
    const { job_id: jobId } = await apiFetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await pollJob(jobId);
    const result = await apiFetch(`/api/jobs/${jobId}/result`);
    renderResultsTable(result.rows);
    const downloadLink = $("download-csv");
    downloadLink.href = `/api/jobs/${jobId}/result.csv`;
    downloadLink.classList.remove("hidden");
  } catch (err) {
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
  $("file-input-x").addEventListener("change", (e) => {
    if (e.target.files[0]) uploadDataset("x", e.target.files[0]);
  });
  $("file-input-y").addEventListener("change", (e) => {
    if (e.target.files[0]) uploadDataset("y", e.target.files[0]);
  });
  $("ko-gene-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addKoGeneFromInput();
    }
  });
  $("run-form").addEventListener("submit", runJob);

  setWorkflow("net");
}

init();
