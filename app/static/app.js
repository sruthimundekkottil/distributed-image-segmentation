// Frontend for the segmentation service. Plain JavaScript, no build step.
//
// Flow:
//   1. the user picks or drops images
//   2. "Process" sends them all to POST /batches (one job per image)
//   3. we poll GET /batches/{id} to update the progress bar and each card's status
//   4. when a job finishes we fetch GET /jobs/{id}/result (COCO JSON with ALL
//      detections) and draw boxes, masks and labels on a <canvas>
//   5. the confidence slider re-draws every canvas locally, no new requests
//   6. "Download COCO JSON" calls GET /batches/{id}/export?min_score=<slider>

"use strict";

const POLL_BATCH_MS = 800;
const POLL_METRICS_MS = 2000;

const $ = (id) => document.getElementById(id);

const state = {
  files: [],          // File objects chosen by the user, not yet sent
  batchId: null,
  items: [],          // one entry per image in the current batch (see createCard)
  minScore: Number($("score-slider").value),
  sliderTouched: false,
};

// ---------------------------------------------------------------- messages

function showError(text) {
  $("error-box").textContent = text;
  $("error-box").hidden = false;
}

function clearError() {
  $("error-box").hidden = true;
}

// Turn a failed fetch response into a readable message.
async function describeError(response) {
  let detail = "";
  try {
    detail = (await response.json()).detail;
  } catch {
    detail = response.statusText;
  }
  if (response.status === 429) {
    // Backpressure: the API's message already says how full the queue is.
    return `The server is busy (HTTP 429). ${detail}\nYour selection was kept, so you can retry.`;
  }
  return `Request failed (HTTP ${response.status}): ${detail}`;
}

// ---------------------------------------------------------------- file selection

function addFiles(fileList) {
  const images = [...fileList].filter((f) => f.type.startsWith("image/"));
  const skipped = fileList.length - images.length;
  state.files.push(...images);
  if (skipped > 0) showError(`Skipped ${skipped} file(s) that are not images.`);
  updateSelection();
}

function updateSelection() {
  const count = state.files.length;
  const sizeMb = state.files.reduce((sum, f) => sum + f.size, 0) / 1e6;
  $("selection-text").textContent = count
    ? `${count} image${count === 1 ? "" : "s"} selected (${sizeMb.toFixed(1)} MB)`
    : "No images selected.";
  $("process-button").disabled = count === 0;
  $("clear-button").disabled = count === 0;
}

const dropZone = $("drop-zone");
$("file-input").addEventListener("change", (event) => {
  addFiles(event.target.files);
  event.target.value = ""; // allow picking the same file again later
});
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault(); // required, otherwise the browser opens the file
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  addFiles(event.dataTransfer.files);
});
$("clear-button").addEventListener("click", () => {
  state.files = [];
  clearError();
  updateSelection();
});

// ---------------------------------------------------------------- submit + poll

$("process-button").addEventListener("click", processBatch);

async function processBatch() {
  clearError();
  const files = state.files;
  const form = new FormData();
  files.forEach((file) => form.append("files", file, file.name));

  setBusy(true);
  let response;
  try {
    response = await fetch("/batches", { method: "POST", body: form });
  } catch {
    showError("Could not reach the server. Is `docker compose up` running?");
    setBusy(false);
    return;
  }
  if (!response.ok) {
    showError(await describeError(response));
    setBusy(false); // keep the selection so the user can retry
    return;
  }

  const batch = await response.json();
  state.batchId = batch.batch_id;
  state.files = [];
  updateSelection();

  // The API returns job IDs in the same order as the uploaded files.
  $("gallery").replaceChildren();
  state.items = batch.job_ids.map((jobId, i) => createCard(jobId, files[i]));
  $("results").hidden = false;
  $("download-button").disabled = true;
  updateProgress({ total: batch.total, completed: 0, failed: 0 });
  pollBatch(batch.batch_id);
}

function setBusy(busy) {
  $("process-button").disabled = busy || state.files.length === 0;
  $("process-button").textContent = busy ? "Processing…" : "Process";
}

async function pollBatch(batchId) {
  if (batchId !== state.batchId) return; // a newer batch replaced this one

  let batch;
  try {
    const response = await fetch(`/batches/${batchId}`);
    if (!response.ok) {
      showError(await describeError(response));
      setBusy(false);
      return;
    }
    batch = await response.json();
  } catch {
    showError("Lost connection to the server. Retrying…");
    setTimeout(() => pollBatch(batchId), POLL_BATCH_MS * 3);
    return;
  }

  updateProgress(batch);
  batch.jobs.forEach((job, i) => updateCard(state.items[i], job));

  if (!batch.done) {
    setTimeout(() => pollBatch(batchId), POLL_BATCH_MS);
    return;
  }
  setBusy(false);
  if (batch.failed > 0) {
    showError(
      `${batch.failed} of ${batch.total} image(s) failed. ` +
        "See the red cards below for the error message from the worker."
    );
  } else {
    clearError();
  }
}

function updateProgress({ total, completed, failed }) {
  const done = completed + failed;
  $("progress").hidden = false;
  $("progress-bar").style.width = `${total ? (100 * done) / total : 0}%`;
  $("progress-bar").classList.toggle("has-failures", failed > 0);
  $("progress-text").textContent =
    `${done} of ${total} processed` +
    (failed ? ` (${completed} succeeded, ${failed} failed)` : "") +
    (done === total ? "." : "…");
}

// ---------------------------------------------------------------- result cards

function createCard(jobId, file) {
  const card = $("result-template").content.firstElementChild.cloneNode(true);
  const item = {
    jobId,
    card,
    status: "queued",
    result: null,
    image: card.querySelector(".original"),
    canvas: card.querySelector(".output"),
    caption: card.querySelector(".output-caption"),
    badge: card.querySelector(".badge"),
    list: card.querySelector(".detections"),
  };
  card.querySelector(".file-name").textContent = file.name;
  card.querySelector(".file-name").title = file.name;
  item.image.alt = file.name;
  setBadge(item, "queued");

  // Show the original immediately from the local file; no need to download it back.
  // `loaded` resolves to true once the browser has the pixels (false if it can't
  // display the file), so we never try to draw on the canvas too early.
  item.loaded = new Promise((resolve) => {
    item.image.addEventListener("load", () => resolve(true), { once: true });
    item.image.addEventListener("error", () => resolve(false), { once: true });
  });
  item.loaded.then((ok) => ok && drawPending(item));
  item.image.src = URL.createObjectURL(file);

  $("gallery").append(card);
  return item;
}

function setBadge(item, status) {
  item.badge.textContent = status;
  item.badge.className = `badge ${status}`;
}

async function updateCard(item, job) {
  if (item.status === job.status) return;
  item.status = job.status;
  setBadge(item, job.status);

  if (job.status === "started") {
    item.caption.textContent = "Running the model…";
  } else if (job.status === "failed" || job.status === "expired") {
    item.caption.textContent = `Failed: ${job.error || "result expired"}`;
    item.caption.classList.add("error");
  } else if (job.status === "finished") {
    item.caption.textContent = "Loading result…";
    const response = await fetch(`/jobs/${item.jobId}/result`);
    if (!response.ok) {
      item.caption.textContent = await describeError(response);
      item.caption.classList.add("error");
      return;
    }
    item.result = await response.json();
    adoptDefaultThreshold(item.result);
    $("download-button").disabled = false;
    if (await item.loaded) {
      render(item);
    } else {
      item.caption.textContent = "Result ready, but this browser can't display the image file.";
    }
  }
}

// Start the slider at the server's preview threshold, unless the user moved it.
function adoptDefaultThreshold(result) {
  const serverDefault = result.info && result.info.preview_min_score;
  if (state.sliderTouched || typeof serverDefault !== "number") return;
  state.minScore = serverDefault;
  $("score-slider").value = serverDefault;
  $("score-value").textContent = serverDefault.toFixed(2);
}

// ---------------------------------------------------------------- canvas drawing

// One distinct colour per class id (DeepLabV3 has 21 classes).
// `lightness` is lowered for label tags so white text stays readable on every hue.
function classColor(categoryId, alpha = 1, lightness = 50) {
  return `hsla(${Math.round((categoryId * 360) / 21)}, 85%, ${lightness}%, ${alpha})`;
}

function drawPending(item) {
  const { canvas, image } = item;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.globalAlpha = 0.35; // faded until the result arrives
  ctx.drawImage(image, 0, 0);
  ctx.globalAlpha = 1;
}

function render(item) {
  const { canvas, image, result } = item;
  const ctx = canvas.getContext("2d");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  ctx.drawImage(image, 0, 0);

  if (!result.annotations) {
    // Not a segmentation result (e.g. MODEL_NAME=mobilenet_v2): show the top label.
    const top = (result.predictions || [])[0];
    item.caption.textContent = top
      ? `Classifier output: ${top.label} (${top.score.toFixed(2)})`
      : "Finished (no boxes for this model)";
    return;
  }

  // The canvas has the image's full resolution but is shown smaller on screen.
  // Sizing lines and text in *screen* pixels keeps labels readable for big
  // and small images alike (14px text, 2px lines as you actually see them).
  const shrink = Math.max(
    canvas.width / (canvas.clientWidth || canvas.width),
    canvas.height / (canvas.clientHeight || canvas.height)
  );
  const lineWidth = 2 * shrink;
  const fontSize = Math.round(13 * shrink);
  const visible = result.annotations.filter((a) => a.score >= state.minScore);

  for (const a of visible) {
    // Translucent mask from the COCO polygons ([x1, y1, x2, y2, ...]).
    ctx.fillStyle = classColor(a.category_id, 0.35);
    for (const polygon of a.segmentation) {
      ctx.beginPath();
      ctx.moveTo(polygon[0], polygon[1]);
      for (let i = 2; i < polygon.length; i += 2) ctx.lineTo(polygon[i], polygon[i + 1]);
      ctx.closePath();
      ctx.fill();
    }
  }

  for (const a of visible) {
    const [x, y, w, h] = a.bbox;
    ctx.strokeStyle = classColor(a.category_id);
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(x, y, w, h);

    // Label tag above the box (or inside it if the box touches the top edge).
    const text = `${a.category_name} ${a.score.toFixed(2)}`;
    ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
    const tagWidth = ctx.measureText(text).width + 10;
    const tagHeight = fontSize + 8;
    const tagY = y - tagHeight >= 0 ? y - tagHeight : y;
    ctx.fillStyle = classColor(a.category_id, 1, 32);
    ctx.fillRect(x, tagY, tagWidth, tagHeight);
    ctx.fillStyle = "#fff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + 5, tagY + tagHeight / 2);
  }

  const total = result.annotations.length;
  item.caption.textContent =
    `${visible.length} of ${total} detection${total === 1 ? "" : "s"} shown ` +
    `(min confidence ${state.minScore.toFixed(2)})`;
  renderDetectionList(item);
}

// Chips under each card: every detection, crossed out if the slider hides it.
function renderDetectionList(item) {
  const chips = [...item.result.annotations]
    .sort((a, b) => b.score - a.score)
    .map((a) => {
      const li = document.createElement("li");
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = classColor(a.category_id);
      li.append(swatch, `${a.category_name} ${a.score.toFixed(2)}`);
      li.classList.toggle("hidden-by-filter", a.score < state.minScore);
      return li;
    });
  item.list.replaceChildren(...chips);
}

// ---------------------------------------------------------------- slider + download

$("score-slider").addEventListener("input", (event) => {
  state.sliderTouched = true;
  state.minScore = Number(event.target.value);
  $("score-value").textContent = state.minScore.toFixed(2);
  state.items.filter((item) => item.result && item.image.naturalWidth).forEach(render);
});

// Label sizes depend on how big the canvas is on screen, so redraw after resizing.
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    state.items.filter((item) => item.result && item.image.naturalWidth).forEach(render);
  }, 150);
});

$("download-button").addEventListener("click", async () => {
  if (!state.batchId) return;
  const url = `/batches/${state.batchId}/export?min_score=${state.minScore}`;
  const response = await fetch(url);
  if (!response.ok) {
    showError(await describeError(response));
    return;
  }
  // Save the JSON through a temporary link so we can show errors (a plain
  // link would navigate away to an error page instead).
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `batch-${state.batchId}.coco.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

// ---------------------------------------------------------------- status panel

// Read one value from Prometheus text format, e.g. "inference_queue_depth 3.0".
function metricValue(text, name) {
  const match = text.match(new RegExp(`^${name} ([0-9.eE+-]+)$`, "m"));
  return match ? Number(match[1]) : null;
}

async function refreshStatus() {
  try {
    const text = await (await fetch("/metrics")).text();
    const show = (id, value) => ($(id).textContent = value === null ? "–" : value);
    show("stat-queue", metricValue(text, "inference_queue_depth"));
    show("stat-completed", metricValue(text, "inference_jobs_completed_total"));
    show("stat-workers", metricValue(text, "inference_workers"));
  } catch {
    // Server unreachable; keep the last values. Errors are shown elsewhere.
  }
  setTimeout(refreshStatus, POLL_METRICS_MS);
}

async function checkModel() {
  try {
    const health = await (await fetch("/health")).json();
    $("stat-model").textContent = health.model || "–";
    if (health.model && health.model !== "deeplabv3_mobilenet") {
      $("warning-box").textContent =
        `The server is running MODEL_NAME=${health.model}, which does not draw boxes. ` +
        "Restart it with MODEL_NAME=deeplabv3_mobilenet for segmentation.";
      $("warning-box").hidden = false;
    }
    if (health.workers === 0) {
      $("warning-box").textContent =
        "No workers are running, so jobs will wait in the queue. " +
        "Start some with: docker compose up --scale worker=2";
      $("warning-box").hidden = false;
    }
  } catch {
    showError("Could not reach the server. Is `docker compose up` running?");
  }
}

checkModel();
refreshStatus();
