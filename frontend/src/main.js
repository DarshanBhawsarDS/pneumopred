/**
 * PneumoPred Medical AI Frontend Client
 * Uncertainty-Aware Chest Radiograph Analysis Platform
 */

// State Management
const state = {
  backendOnline: false,
  selectedFile: null,
  selectedFileBlob: null,
  previewDataUrl: null,
  mcSamples: 20,
  overlayAlpha: 0.45,
  currentView: 'overlay', // 'overlay' | 'heatmap' | 'original'
  analysisResult: null,
};

// DOM References
const elements = {
  statusPill: document.getElementById('backendStatusPill'),
  statusText: document.getElementById('backendStatusText'),
  devicePill: document.getElementById('devicePill'),
  deviceText: document.getElementById('deviceText'),
  
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  dropzonePrompt: document.getElementById('dropzonePrompt'),
  previewWrapper: document.getElementById('previewWrapper'),
  imagePreview: document.getElementById('imagePreview'),
  btnClearImage: document.getElementById('btnClearImage'),
  previewMeta: document.getElementById('previewMeta'),
  
  btnSampleNormal: document.getElementById('btnSampleNormal'),
  btnSamplePneumonia: document.getElementById('btnSamplePneumonia'),
  
  mcButtons: document.querySelectorAll('.mc-btn'),
  alphaSlider: document.getElementById('alphaSlider'),
  alphaValDisplay: document.getElementById('alphaValDisplay'),
  
  btnAnalyze: document.getElementById('btnAnalyze'),
  btnSpinner: document.getElementById('btnSpinner'),
  btnText: document.getElementById('btnText'),
  
  emptyState: document.getElementById('emptyState'),
  resultsContent: document.getElementById('resultsContent'),
  inferenceTelemetry: document.getElementById('inferenceTelemetry'),
  
  verdictBanner: document.getElementById('verdictBanner'),
  verdictValue: document.getElementById('verdictValue'),
  verdictSub: document.getElementById('verdictSub'),
  confidencePercent: document.getElementById('confidencePercent'),
  
  probNormalVal: document.getElementById('probNormalVal'),
  probNormalBar: document.getElementById('probNormalBar'),
  probPneumoniaVal: document.getElementById('probPneumoniaVal'),
  probPneumoniaBar: document.getElementById('probPneumoniaBar'),
  
  viewButtons: document.querySelectorAll('.view-btn'),
  gradcamDisplayImg: document.getElementById('gradcamDisplayImg'),
  gradcamTargetLayer: document.getElementById('gradcamTargetLayer'),
  saliencyExplanation: document.getElementById('saliencyExplanation'),
  
  samplesBadge: document.getElementById('samplesBadge'),
  valEntropy: document.getElementById('valEntropy'),
  entropyBadge: document.getElementById('entropyBadge'),
  valAleatoric: document.getElementById('valAleatoric'),
  valEpistemic: document.getElementById('valEpistemic'),
  
  decisionSupportBox: document.getElementById('decisionSupportBox'),
  decisionFlag: document.getElementById('decisionFlag'),
  decisionDesc: document.getElementById('decisionDesc'),
  
  footerModel: document.getElementById('footerModel'),
  footerDevice: document.getElementById('footerDevice'),
  footerLatency: document.getElementById('footerLatency'),
};

/* ---------------- BACKEND HEALTH MONITOR ---------------- */
async function checkBackendHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    
    state.backendOnline = true;
    elements.statusPill.className = 'status-pill connected';
    
    if (data.model_loaded) {
      elements.statusText.textContent = `Online (${data.model || 'EfficientNet-B0'})`;
    } else {
      elements.statusText.textContent = 'Model Loading / Standby';
    }
    
    if (data.device) {
      elements.devicePill.style.display = 'block';
      elements.deviceText.textContent = data.device.toUpperCase();
    }
  } catch (err) {
    state.backendOnline = false;
    elements.statusPill.className = 'status-pill disconnected';
    elements.statusText.textContent = 'Backend Offline (Check port 8000)';
    elements.devicePill.style.display = 'none';
  }
}

/* ---------------- FILE HANDLING & PREVIEW ---------------- */
function setFile(file, dataUrl = null) {
  state.selectedFile = file;
  elements.btnAnalyze.disabled = false;
  elements.dropzonePrompt.style.display = 'none';
  elements.previewWrapper.style.display = 'flex';
  elements.previewMeta.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

  if (dataUrl) {
    state.previewDataUrl = dataUrl;
    elements.imagePreview.src = dataUrl;
  } else {
    const reader = new FileReader();
    reader.onload = (e) => {
      state.previewDataUrl = e.target.result;
      elements.imagePreview.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
}

function clearFile() {
  state.selectedFile = null;
  state.previewDataUrl = null;
  elements.fileInput.value = '';
  elements.previewWrapper.style.display = 'none';
  elements.dropzonePrompt.style.display = 'flex';
  elements.btnAnalyze.disabled = true;
}

// Drag & drop listeners
elements.dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  elements.dropzone.classList.add('drag-active');
});

elements.dropzone.addEventListener('dragleave', () => {
  elements.dropzone.classList.remove('drag-active');
});

elements.dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  elements.dropzone.classList.remove('drag-active');
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    const file = e.dataTransfer.files[0];
    if (file.type.startsWith('image/')) {
      setFile(file);
    } else {
      alert('Please upload a valid radiograph image (JPEG or PNG).');
    }
  }
});

elements.fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files.length > 0) {
    setFile(e.target.files[0]);
  }
});

elements.btnClearImage.addEventListener('click', (e) => {
  e.stopPropagation();
  clearFile();
});

/* ---------------- SYNTHETIC X-RAY GENERATOR (FOR INSTANT SAMPLES) ---------------- */
function generateChestXRaySample(type = 'normal') {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  // Dark thoracic cavity background
  ctx.fillStyle = '#0a0d14';
  ctx.fillRect(0, 0, 512, 512);

  // Rib cage / soft tissue gradient background
  const bgGrad = ctx.createRadialGradient(256, 256, 50, 256, 256, 260);
  bgGrad.addColorStop(0, '#1c2230');
  bgGrad.addColorStop(0.7, '#11151f');
  bgGrad.addColorStop(1, '#05070a');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(20, 20, 472, 472);

  // Left & Right Lung Fields (translucent darker lung aeration)
  const drawLung = (cx, cy, w, h, angle) => {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.ellipse(0, 0, w, h, 0, 0, Math.PI * 2);
    ctx.fillStyle = '#080a0e';
    ctx.fill();
    ctx.lineWidth = 14;
    ctx.strokeStyle = '#2d3748';
    ctx.stroke();
    ctx.restore();
  };

  // Right Lung Field (anatomical right = viewer left)
  drawLung(170, 240, 75, 140, -0.05);
  // Left Lung Field (anatomical left = viewer right)
  drawLung(342, 240, 75, 140, 0.05);

  // Mediastinum / Spine central column
  ctx.fillStyle = '#4a5568';
  ctx.fillRect(240, 60, 32, 400);

  // Heart silhouette (Cardiac shadow shifted left)
  ctx.beginPath();
  ctx.ellipse(280, 290, 65, 80, -0.3, 0, Math.PI * 2);
  ctx.fillStyle = '#3e4856';
  ctx.fill();

  // Clavicles
  ctx.lineWidth = 8;
  ctx.strokeStyle = '#718096';
  ctx.beginPath();
  ctx.moveTo(100, 110);
  ctx.quadraticCurveTo(180, 130, 240, 120);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(412, 110);
  ctx.quadraticCurveTo(332, 130, 272, 120);
  ctx.stroke();

  // Rib arches
  ctx.lineWidth = 4;
  ctx.strokeStyle = 'rgba(160, 174, 192, 0.45)';
  for (let y = 160; y <= 380; y += 38) {
    ctx.beginPath();
    ctx.moveTo(100, y);
    ctx.quadraticCurveTo(180, y - 20, 240, y - 10);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(412, y);
    ctx.quadraticCurveTo(332, y - 20, 272, y - 10);
    ctx.stroke();
  }

  // If Pneumonia, add dense parenchymal alveolar infiltrate / opacity
  if (type === 'pneumonia') {
    const infGrad = ctx.createRadialGradient(165, 270, 10, 165, 270, 80);
    infGrad.addColorStop(0, 'rgba(230, 235, 245, 0.88)');
    infGrad.addColorStop(0.5, 'rgba(200, 210, 225, 0.65)');
    infGrad.addColorStop(1, 'rgba(120, 130, 150, 0.0)');
    ctx.fillStyle = infGrad;
    ctx.beginPath();
    ctx.arc(165, 270, 80, 0, Math.PI * 2);
    ctx.fill();

    // Patchy consolidation scatter
    for (let i = 0; i < 30; i++) {
      const rx = 140 + Math.random() * 60;
      const ry = 230 + Math.random() * 80;
      const rr = 6 + Math.random() * 14;
      ctx.fillStyle = 'rgba(240, 245, 255, 0.4)';
      ctx.beginPath();
      ctx.arc(rx, ry, rr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      const filename = type === 'pneumonia' ? 'sample_pneumonia_infiltrate.jpg' : 'sample_normal_thorax.jpg';
      const file = new File([blob], filename, { type: 'image/jpeg' });
      resolve({ file, dataUrl: canvas.toDataURL('image/jpeg') });
    }, 'image/jpeg', 0.95);
  });
}

elements.btnSampleNormal.addEventListener('click', async () => {
  const sample = await generateChestXRaySample('normal');
  setFile(sample.file, sample.dataUrl);
});

elements.btnSamplePneumonia.addEventListener('click', async () => {
  const sample = await generateChestXRaySample('pneumonia');
  setFile(sample.file, sample.dataUrl);
});

/* ---------------- PARAMETER CONTROLS ---------------- */
elements.mcButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    elements.mcButtons.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    state.mcSamples = parseInt(btn.dataset.samples, 10);
  });
});

elements.alphaSlider.addEventListener('input', (e) => {
  state.overlayAlpha = parseFloat(e.target.value);
  elements.alphaValDisplay.textContent = state.overlayAlpha.toFixed(2);
});

/* ---------------- GRAD-CAM VIEW SWITCHER ---------------- */
elements.viewButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    elements.viewButtons.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    state.currentView = btn.dataset.view;
    updateGradcamDisplay();
  });
});

function updateGradcamDisplay() {
  if (!state.analysisResult || !state.analysisResult.explanation) return;
  const exp = state.analysisResult.explanation;
  
  if (state.currentView === 'overlay') {
    elements.gradcamDisplayImg.src = `data:image/jpeg;base64,${exp.overlay_base64}`;
  } else if (state.currentView === 'heatmap') {
    elements.gradcamDisplayImg.src = `data:image/jpeg;base64,${exp.heatmap_base64}`;
  } else if (state.currentView === 'original') {
    elements.gradcamDisplayImg.src = state.previewDataUrl;
  }
}

/* ---------------- INFERENCE DISPATCH ---------------- */
elements.btnAnalyze.addEventListener('click', async () => {
  if (!state.selectedFile) return;

  // Set loading UI
  elements.btnAnalyze.disabled = true;
  elements.btnSpinner.style.display = 'inline-block';
  elements.btnText.textContent = 'Running MC Dropout & Grad-CAM...';
  elements.inferenceTelemetry.textContent = 'Analyzing...';

  try {
    const formData = new FormData();
    formData.append('file', state.selectedFile);

    const queryParams = new URLSearchParams({
      include_uncertainty: 'true',
      include_explainability: 'true',
      num_samples: state.mcSamples.toString(),
    });

    const response = await fetch(`/api/analyze?${queryParams.toString()}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => null);
      throw new Error(errJson?.detail || `Inference error: HTTP ${response.status}`);
    }

    const data = await response.json();
    state.analysisResult = data;
    renderResults(data);

  } catch (err) {
    console.error('Inference failed:', err);
    alert(`Diagnostics failed: ${err.message}`);
  } finally {
    elements.btnAnalyze.disabled = false;
    elements.btnSpinner.style.display = 'none';
    elements.btnText.textContent = 'Run Diagnostic Inference';
  }
});

/* ---------------- RENDER ANALYSIS FINDINGS ---------------- */
function renderResults(data) {
  elements.emptyState.style.display = 'none';
  elements.resultsContent.style.display = 'flex';

  const isNormal = data.predicted_class === 'NORMAL';
  const confPercent = (data.confidence * 100).toFixed(1);

  // Verdict Banner
  elements.verdictValue.textContent = data.predicted_class;
  elements.confidencePercent.textContent = `${confPercent}%`;

  if (isNormal) {
    elements.verdictBanner.className = 'verdict-banner is-normal';
    elements.verdictSub.textContent = 'No acute pulmonary consolidation detected';
  } else {
    elements.verdictBanner.className = 'verdict-banner';
    elements.verdictSub.textContent = 'Features consistent with pulmonary consolidation / pneumonia';
  }

  // Probability Distributions
  const pNorm = ((data.probabilities?.NORMAL || 0) * 100).toFixed(1);
  const pPneu = ((data.probabilities?.PNEUMONIA || 0) * 100).toFixed(1);

  elements.probNormalVal.textContent = `${pNorm}%`;
  elements.probNormalBar.style.width = `${pNorm}%`;
  elements.probPneumoniaVal.textContent = `${pPneu}%`;
  elements.probPneumoniaBar.style.width = `${pPneu}%`;

  // Explainability
  if (data.explanation) {
    elements.gradcamTargetLayer.textContent = data.explanation.target_layer || 'backbone.conv_head';
    elements.viewButtons.forEach((b) => b.classList.remove('active'));
    document.querySelector('.view-btn[data-view="overlay"]').classList.add('active');
    state.currentView = 'overlay';
    updateGradcamDisplay();
  }

  // Uncertainty Quantification
  if (data.uncertainty) {
    const unc = data.uncertainty;
    elements.samplesBadge.textContent = `${unc.num_samples} Stochastic Passes`;
    
    // Entropy
    const entropyVal = unc.predictive_entropy.toFixed(3);
    elements.valEntropy.innerHTML = `${entropyVal} <span class="unit">bits</span>`;
    
    if (unc.predictive_entropy < 0.20) {
      elements.entropyBadge.className = 'metric-badge';
      elements.entropyBadge.textContent = 'Low';
    } else if (unc.predictive_entropy < 0.50) {
      elements.entropyBadge.className = 'metric-badge medium';
      elements.entropyBadge.textContent = 'Moderate';
    } else {
      elements.entropyBadge.className = 'metric-badge high';
      elements.entropyBadge.textContent = 'Elevated';
    }

    elements.valAleatoric.innerHTML = `${unc.aleatoric_expected_entropy.toFixed(3)} <span class="unit">bits</span>`;
    elements.valEpistemic.innerHTML = `${unc.epistemic_mutual_information.toFixed(3)} <span class="unit">bits</span>`;
  }

  // Decision Support
  if (data.decision_support) {
    const ds = data.decision_support;
    if (ds.status === 'CONFIDENT') {
      elements.decisionSupportBox.className = 'decision-support-box';
      elements.decisionFlag.textContent = 'DECISION SUPPORT: CONFIDENT';
      elements.decisionDesc.textContent = ds.recommendation || 'Low predictive entropy across stochastic dropout passes.';
    } else {
      elements.decisionSupportBox.className = 'decision-support-box uncertain';
      elements.decisionFlag.textContent = 'DECISION SUPPORT: UNCERTAIN / BORDERLINE';
      elements.decisionDesc.textContent = ds.recommendation || 'Elevated entropy detected. Further radiological review recommended.';
    }
  }

  // Telemetry Footer
  elements.footerModel.textContent = `Model: ${data.model || 'EfficientNet-B0'}`;
  elements.footerDevice.textContent = `Compute: ${data.device || 'CPU'}`;
  elements.footerLatency.textContent = `Latency: ${data.inference_time_ms} ms`;
  elements.inferenceTelemetry.textContent = `Completed in ${data.inference_time_ms} ms`;
}

/* ---------------- INITIALIZATION ---------------- */
checkBackendHealth();
setInterval(checkBackendHealth, 8000);
