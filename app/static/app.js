/* Weather Classifier — Frontend Logic */

let selectedFile = null;

/**
 * Handle file selection from the upload input.
 * Switches from upload zone to preview, enables classify button.
 */
function handleFile(event) {
  selectedFile = event.target.files[0];
  if (!selectedFile) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('preview').src = e.target.result;
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('previewContainer').style.display = 'block';
    document.getElementById('fileName').textContent =
      selectedFile.name.length > 22
        ? selectedFile.name.slice(0, 20) + '…'
        : selectedFile.name;
  };
  reader.readAsDataURL(selectedFile);

  document.getElementById('classifyBtn').disabled = false;
  document.getElementById('resultsContent').style.display = 'none';
  document.getElementById('emptyState').style.display = 'flex';
}

/**
 * Handle drag-and-drop onto the upload zone.
 */
function handleDrop(event) {
  event.preventDefault();
  handleDragLeave();
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const input = document.getElementById('fileInput');
  const dt = new DataTransfer();
  dt.items.add(file);
  input.files = dt.files;
  handleFile({ target: input });
}

function handleDragOver(event) {
  event.preventDefault();
  document.getElementById('uploadZone').style.borderColor = 'var(--text-info)';
  document.getElementById('uploadZone').style.background = 'var(--bg-info)';
}

function handleDragLeave() {
  document.getElementById('uploadZone').style.borderColor = '';
  document.getElementById('uploadZone').style.background = '';
}

/**
 * Send the selected image to /predict and render results.
 */
async function classify() {
  if (!selectedFile) return;

  const btn = document.getElementById('classifyBtn');
  btn.disabled = true;
  btn.innerHTML = '<i class="ti ti-loader-2" aria-hidden="true"></i> Analyzing...';

  const form = new FormData();
  form.append('file', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    btn.innerHTML = '<i class="ti ti-alert-circle" aria-hidden="true"></i> Error — try again';
    btn.disabled = false;
    console.error(err);
    return;
  }

  btn.innerHTML = '<i class="ti ti-sparkles" aria-hidden="true"></i> Classify image';
  btn.disabled = false;
}

/**
 * Render prediction results into the DOM.
 *
 * @param {Object} data           - Response from /predict.
 * @param {string} data.prediction  - Predicted class label.
 * @param {number} data.confidence  - Confidence score 0–1.
 * @param {Array}  data.top5        - [{class, confidence}, ...].
 * @param {string} data.heatmap_b64 - Base64-encoded GradCAM PNG.
 */
function renderResults(data) {
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'flex';

  const pct = (data.confidence * 100).toFixed(1);
  document.getElementById('predLabel').textContent = data.prediction;
  document.getElementById('confBadge').textContent = pct + '%';
  document.getElementById('confBar').style.width = pct + '%';

  document.getElementById('bars').innerHTML = data.top5.map((item) => `
    <div class="bar-row">
      <span class="bar-label">${item.class}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${(item.confidence * 100).toFixed(1)}%"></div>
      </div>
      <span class="bar-pct">${(item.confidence * 100).toFixed(1)}%</span>
    </div>`).join('');

  document.getElementById('origImg').src = document.getElementById('preview').src;
  document.getElementById('heatmapImg').src = 'data:image/png;base64,' + data.heatmap_b64;
}
