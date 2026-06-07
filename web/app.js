const state = {
  file: null,
  fileUrl: null,
  videoNaturalWidth: 0,
  videoNaturalHeight: 0,
  roi: { x: 0, y: 0, width: 1, height: 1 },
  isDrawing: false,
  start: null,
  pollTimer: null,
  frames: [],
  selectedFrameIndex: 0,
  frameRequestId: 0,
};

const els = {
  serverStatus: document.querySelector("#serverStatus"),
  videoFile: document.querySelector("#videoFile"),
  videoPreview: document.querySelector("#videoPreview"),
  framePreview: document.querySelector("#framePreview"),
  roiCanvas: document.querySelector("#roiCanvas"),
  frameBrowser: document.querySelector("#frameBrowser"),
  frameStatus: document.querySelector("#frameStatus"),
  frameList: document.querySelector("#frameList"),
  emptyState: document.querySelector("#emptyState"),
  roiX: document.querySelector("#roiX"),
  roiY: document.querySelector("#roiY"),
  roiWidth: document.querySelector("#roiWidth"),
  roiHeight: document.querySelector("#roiHeight"),
  referenceFrame: document.querySelector("#referenceFrame"),
  pipelineMode: document.querySelector("#pipelineMode"),
  removalMode: document.querySelector("#removalMode"),
  sam2TrackingBackend: document.querySelector("#sam2TrackingBackend"),
  resourceProfile: document.querySelector("#resourceProfile"),
  qualityRestoration: document.querySelector("#qualityRestoration"),
  vlmProvider: document.querySelector("#vlmProvider"),
  inpaintEngine: document.querySelector("#inpaintEngine"),
  method: document.querySelector("#method"),
  chunkSize: document.querySelector("#chunkSize"),
  radius: document.querySelector("#radius"),
  maskPadding: document.querySelector("#maskPadding"),
  sam2Bidirectional: document.querySelector("#sam2Bidirectional"),
  runVoidPass: document.querySelector("#runVoidPass"),
  voidShadowDilation: document.querySelector("#voidShadowDilation"),
  heuristicContactDilation: document.querySelector("#heuristicContactDilation"),
  heuristicShadowDilation: document.querySelector("#heuristicShadowDilation"),
  heuristicShadowOffset: document.querySelector("#heuristicShadowOffset"),
  maxChunks: document.querySelector("#maxChunks"),
  processButton: document.querySelector("#processButton"),
  progressLabel: document.querySelector("#progressLabel"),
  progressCount: document.querySelector("#progressCount"),
  progressFill: document.querySelector("#progressFill"),
  downloadLink: document.querySelector("#downloadLink"),
  errorMessage: document.querySelector("#errorMessage"),
  logList: document.querySelector("#logList"),
  logCount: document.querySelector("#logCount"),
};

const canvasContext = els.roiCanvas.getContext("2d");

boot();

function boot() {
  bindEvents();
  checkHealth();
}

function bindEvents() {
  els.videoFile.addEventListener("change", handleFileChange);
  els.videoPreview.addEventListener("loadedmetadata", syncCanvasToVideo);
  window.addEventListener("resize", drawRoi);
  els.roiCanvas.addEventListener("pointerdown", startDrawing);
  els.roiCanvas.addEventListener("pointermove", continueDrawing);
  window.addEventListener("pointerup", stopDrawing);
  els.processButton.addEventListener("click", submitJob);
  els.pipelineMode.addEventListener("change", updatePipelineVisibility);
  els.referenceFrame.addEventListener("input", handleReferenceFrameInput);

  [els.roiX, els.roiY, els.roiWidth, els.roiHeight].forEach((input) => {
    input.addEventListener("input", handleRoiInput);
  });
  updatePipelineVisibility();
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Server health check failed.");
    els.serverStatus.textContent = "Server ready";
    els.serverStatus.classList.add("ready");
  } catch (error) {
    els.serverStatus.textContent = "Server unavailable";
    showError(error.message);
  }
}

function handleFileChange(event) {
  const [file] = event.target.files;
  state.file = file || null;
  state.frames = [];
  state.selectedFrameIndex = 0;
  state.frameRequestId += 1;
  els.downloadLink.classList.add("hidden");
  resetProgress();
  resetLogs();
  resetFrameBrowser();

  if (!state.file) {
    if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
    state.fileUrl = null;
    els.videoPreview.removeAttribute("src");
    els.framePreview.removeAttribute("src");
    els.framePreview.classList.add("hidden");
    els.videoPreview.classList.remove("hidden");
    els.emptyState.classList.remove("hidden");
    return;
  }

  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.fileUrl = URL.createObjectURL(state.file);
  els.videoPreview.src = state.fileUrl;
  els.videoPreview.classList.remove("hidden");
  els.framePreview.classList.add("hidden");
  els.emptyState.classList.add("hidden");
  showError("No errors yet.");
  loadFrameBrowser(state.file, state.frameRequestId);
}

function syncCanvasToVideo() {
  syncCanvasDimensions(els.videoPreview.videoWidth, els.videoPreview.videoHeight);
}

function syncCanvasDimensions(width, height) {
  state.videoNaturalWidth = width;
  state.videoNaturalHeight = height;
  els.roiCanvas.parentElement.style.aspectRatio =
    `${state.videoNaturalWidth} / ${state.videoNaturalHeight}`;
  els.roiCanvas.width = state.videoNaturalWidth;
  els.roiCanvas.height = state.videoNaturalHeight;
  state.roi = defaultRoi();
  writeRoiInputs();
  drawRoi();
}

async function loadFrameBrowser(file, requestId) {
  els.frameBrowser.classList.remove("hidden");
  els.frameStatus.textContent = "Extracting";
  els.frameList.textContent = "";

  const formData = new FormData();
  formData.append("video", file);
  formData.append("thumbnail_width", "320");

  try {
    const response = await fetch("/api/video-frames", { method: "POST", body: formData });
    const payload = await readJson(response);
    if (requestId !== state.frameRequestId) return;
    if (!response.ok) throw new Error(payload.detail || "Could not extract video frames.");

    state.frames = payload.frames || [];
    if (!state.frames.length) throw new Error("No frames were extracted from the video.");
    syncCanvasDimensions(payload.width, payload.height);
    renderFrameBrowser(payload.frame_count || state.frames.length);
    selectFrame(state.frames[0].index);
    els.frameStatus.textContent = `${state.frames.length} frames`;
  } catch (error) {
    if (requestId !== state.frameRequestId) return;
    resetFrameBrowser();
    els.videoPreview.classList.remove("hidden");
    els.framePreview.classList.add("hidden");
    showError(`Frame browser unavailable: ${error.message}`);
  }
}

function renderFrameBrowser(frameCount) {
  const fragment = document.createDocumentFragment();
  state.frames.forEach((frame) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "frame-thumb";
    button.dataset.frameIndex = String(frame.index);
    button.setAttribute("aria-label", `Frame ${frame.index}`);

    const image = document.createElement("img");
    image.src = frame.src;
    image.alt = `Frame ${frame.index}`;
    image.loading = "lazy";

    const label = document.createElement("span");
    label.textContent = `Frame ${frame.index}`;

    button.append(image, label);
    button.addEventListener("click", () => selectFrame(frame.index));
    fragment.append(button);
  });
  els.frameList.replaceChildren(fragment);
  els.frameStatus.textContent = `${frameCount} frames`;
}

function selectFrame(frameIndex) {
  const frame = state.frames.find((candidate) => candidate.index === frameIndex);
  if (!frame) return;

  state.selectedFrameIndex = frame.index;
  els.referenceFrame.value = frame.index;
  els.framePreview.src = frame.src;
  els.framePreview.classList.remove("hidden");
  els.videoPreview.classList.add("hidden");

  els.frameList.querySelectorAll(".frame-thumb").forEach((button) => {
    const isActive = Number(button.dataset.frameIndex) === frame.index;
    button.classList.toggle("active", isActive);
    if (isActive) {
      button.scrollIntoView({ block: "nearest" });
    }
  });
  drawRoi();
}

function handleReferenceFrameInput() {
  const frameIndex = numberValue(els.referenceFrame, 0);
  if (!state.frames.length) return;
  const frame = state.frames.find((candidate) => candidate.index === frameIndex);
  if (frame) selectFrame(frame.index);
}

function resetFrameBrowser() {
  els.frameBrowser.classList.add("hidden");
  els.frameStatus.textContent = "Waiting";
  els.frameList.textContent = "";
}

function defaultRoi() {
  const width = Math.max(1, Math.round(state.videoNaturalWidth * 0.16));
  const height = Math.max(1, Math.round(state.videoNaturalHeight * 0.12));
  return {
    x: Math.max(0, state.videoNaturalWidth - width - 24),
    y: 24,
    width,
    height,
  };
}

function startDrawing(event) {
  if (!state.videoNaturalWidth || !state.videoNaturalHeight) return;
  els.roiCanvas.setPointerCapture(event.pointerId);
  state.isDrawing = true;
  state.start = eventToVideoPoint(event);
  state.roi = { x: state.start.x, y: state.start.y, width: 1, height: 1 };
  writeRoiInputs();
  drawRoi();
}

function continueDrawing(event) {
  if (!state.isDrawing || !state.start) return;
  const point = eventToVideoPoint(event);
  const x1 = Math.min(state.start.x, point.x);
  const y1 = Math.min(state.start.y, point.y);
  const x2 = Math.max(state.start.x, point.x);
  const y2 = Math.max(state.start.y, point.y);
  state.roi = {
    x: x1,
    y: y1,
    width: Math.max(1, x2 - x1),
    height: Math.max(1, y2 - y1),
  };
  writeRoiInputs();
  drawRoi();
}

function stopDrawing() {
  state.isDrawing = false;
  state.start = null;
}

function eventToVideoPoint(event) {
  const rect = els.roiCanvas.getBoundingClientRect();
  const scaleX = state.videoNaturalWidth / rect.width;
  const scaleY = state.videoNaturalHeight / rect.height;
  const x = Math.round((event.clientX - rect.left) * scaleX);
  const y = Math.round((event.clientY - rect.top) * scaleY);
  return {
    x: clamp(x, 0, state.videoNaturalWidth - 1),
    y: clamp(y, 0, state.videoNaturalHeight - 1),
  };
}

function handleRoiInput() {
  state.roi = {
    x: numberValue(els.roiX, 0),
    y: numberValue(els.roiY, 0),
    width: numberValue(els.roiWidth, 1),
    height: numberValue(els.roiHeight, 1),
  };
  drawRoi();
}

function writeRoiInputs() {
  els.roiX.value = state.roi.x;
  els.roiY.value = state.roi.y;
  els.roiWidth.value = state.roi.width;
  els.roiHeight.value = state.roi.height;
}

function drawRoi() {
  canvasContext.clearRect(0, 0, els.roiCanvas.width, els.roiCanvas.height);
  if (!state.videoNaturalWidth || !state.videoNaturalHeight) return;

  const { x, y, width, height } = state.roi;
  canvasContext.fillStyle = "rgba(15, 118, 110, 0.18)";
  canvasContext.strokeStyle = "#eab308";
  canvasContext.lineWidth = Math.max(2, Math.round(state.videoNaturalWidth / 360));
  canvasContext.fillRect(x, y, width, height);
  canvasContext.strokeRect(x, y, width, height);
}

async function submitJob() {
  if (!state.file) {
    showError("Please choose a video file first.");
    return;
  }

  const formData = new FormData();
  formData.append("video", state.file);
  formData.append("x", state.roi.x);
  formData.append("y", state.roi.y);
  formData.append("width", state.roi.width);
  formData.append("height", state.roi.height);
  formData.append("reference_frame", numberValue(els.referenceFrame, 0));
  formData.append("pipeline_mode", els.pipelineMode.value);
  formData.append("removal_mode", els.removalMode.value);
  formData.append("sam2_tracking_backend", els.sam2TrackingBackend.value);
  formData.append("resource_profile", els.resourceProfile.value);
  formData.append("quality_restoration", els.qualityRestoration.value);
  formData.append("vlm_provider", els.vlmProvider.value);
  formData.append("inpaint_engine", els.inpaintEngine.value);
  formData.append("chunk_size", numberValue(els.chunkSize, 30));
  formData.append("method", els.method.value);
  formData.append("radius", numberValue(els.radius, 3));
  formData.append("mask_padding", numberValue(els.maskPadding, 0));
  formData.append("sam2_bidirectional", els.sam2Bidirectional.checked ? "true" : "false");
  formData.append("run_void", els.runVoidPass.checked ? "true" : "false");
  formData.append("void_shadow_dilation_px", numberValue(els.voidShadowDilation, 0));
  formData.append(
    "heuristic_contact_dilation_px",
    numberValue(els.heuristicContactDilation, 10),
  );
  formData.append(
    "heuristic_shadow_dilation_px",
    numberValue(els.heuristicShadowDilation, 30),
  );
  formData.append(
    "heuristic_shadow_vertical_offset_px",
    numberValue(els.heuristicShadowOffset, 16),
  );
  formData.append("force_mask_cache_rebuild", "true");
  if (els.maxChunks.value.trim()) {
    formData.append("max_chunks", numberValue(els.maxChunks, 1));
  }

  setBusy(true);
  resetProgress();
  resetLogs();
  showError("No errors yet.");

  try {
    const response = await fetch("/api/jobs", { method: "POST", body: formData });
    const payload = await readJson(response);
    if (!response.ok) throw new Error(payload.detail || "Could not start processing.");
    pollJob(payload.job_id);
  } catch (error) {
    setBusy(false);
    showError(error.message);
  }
}

function pollJob(jobId) {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    try {
      const response = await fetch(`/api/jobs/${jobId}`);
      const payload = await readJson(response);
      if (!response.ok) throw new Error(payload.detail || "Could not read job status.");
      renderJobStatus(payload);
    } catch (error) {
      clearInterval(state.pollTimer);
      setBusy(false);
      showError(error.message);
    }
  }, 900);
}

function renderJobStatus(job) {
  const total = job.total_frames || 0;
  const current = job.current_frame || 0;
  let percent = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;

  if (job.state === "running" && job.phase === "finalizing") {
    percent = Math.min(percent, 99);
  }

  els.progressLabel.textContent = statusLabel(job.state, job.phase);
  els.progressCount.textContent = progressText(job, percent, current, total);
  els.progressFill.style.width = `${percent}%`;
  renderLogs(job.logs || []);

  if (job.state === "completed") {
    clearInterval(state.pollTimer);
    setBusy(false);
    els.progressFill.style.width = "100%";
    els.progressCount.textContent = "100%";
    els.downloadLink.href = job.download_url;
    els.downloadLink.classList.remove("hidden");
    renderLogs(job.logs || []);
  }

  if (job.state === "failed") {
    clearInterval(state.pollTimer);
    setBusy(false);
    showError(job.error || "Processing failed.");
  }
}

function statusLabel(status, phase) {
  if (status === "queued") return "Queued";
  if (status === "running" && phase === "preparing") return "Preparing";
  if (status === "running" && phase === "object_detection") return "Object selection";
  if (status === "running" && phase === "sampling") return "Sampling frame";
  if (status === "running" && phase === "masking") return "Masking target";
  if (status === "running" && phase === "segmenting") return "Segmenting target";
  if (status === "running" && phase === "tracking") return "Tracking target";
  if (status === "running" && phase === "chunking") return "Chunking for VOID";
  if (status === "running" && phase === "void_pass") return "VOID pass";
  if (status === "running" && phase === "merging") return "Merging output";
  if (status === "running" && phase === "inpainting") return "Inpainting";
  if (status === "running" && phase === "finalizing") return "Finalizing video";
  if (status === "running") return "Processing";
  if (status === "completed") return "Completed";
  if (status === "failed") return "Failed";
  return "Waiting";
}

function progressText(job, percent, current, total) {
  if (job.state === "running" && job.phase === "finalizing") {
    return "Finalizing";
  }
  if (total > 0) {
    return `${percent}%`;
  }
  return `${current} frames`;
}

function setBusy(isBusy) {
  els.processButton.disabled = isBusy;
  els.processButton.textContent = isBusy ? "Processing" : "Process Video";
}

function resetProgress() {
  clearInterval(state.pollTimer);
  els.progressLabel.textContent = "Waiting";
  els.progressCount.textContent = "0%";
  els.progressFill.style.width = "0%";
}

function resetLogs() {
  els.logCount.textContent = "0";
  els.logList.textContent = "Waiting for job logs.";
}

function renderLogs(logs) {
  els.logCount.textContent = String(logs.length);
  if (!logs.length) {
    els.logList.textContent = "Waiting for job logs.";
    return;
  }
  els.logList.textContent = logs
    .map((entry) => {
      const phase = entry.phase ? ` ${entry.phase}` : "";
      return `[${formatLogTime(entry.time)}${phase}] ${entry.message}`;
    })
    .join("\n");
  els.logList.scrollTop = els.logList.scrollHeight;
}

function formatLogTime(value) {
  if (!value) return "--:--:--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--:--:--";
  return date.toLocaleTimeString([], { hour12: false });
}

function updatePipelineVisibility() {
  const isVoid = els.pipelineMode.value === "void";
  document.querySelectorAll(".void-settings").forEach((element) => {
    element.classList.toggle("hidden", !isVoid);
  });
}

function showError(message) {
  els.errorMessage.textContent = message;
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

function numberValue(input, fallback) {
  const value = Number(input.value);
  return Number.isFinite(value) ? value : fallback;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
