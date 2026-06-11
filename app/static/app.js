/* Weather Classifier — Frontend Logic */

let selectedFile = null;

/**
 * Handle file selection from the upload input.
 * Shows a preview image and enables the classify button.
 */
function handleFile(event) {
  selectedFile = event.target.files[0];
  if (!selectedFile) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    const preview = document.getElementById("preview");
    preview.src = e.target.result;
    preview.style.display = "block";
  };
  reader.readAsDataURL(selectedFile);

  document.getElementById("classifyBtn").disabled = false;
  document.getElementById("results").style.display = "none";
}

/**
 * Handle drag-and-drop onto the upload area.
 */
function handleDrop(event) {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const input = document.getElementById("fileInput");
  const dt = new DataTransfer();
  dt.items.add(file);
  input.files = dt.files;
  handleFile({ target: input });
}

function handleDragOver(event) {
  event.preventDefault();
  document.querySelector(".upload-area").style.borderColor = "#6c8cf5";
}

function handleDragLeave() {
  document.querySelector(".upload-area").style.borderColor = "#3a3f55";
}

/**
 * Send the selected image to /predict and render results.
 */
async function classify() {
  if (!selectedFile) return;

  // Show loading state
  const results = document.getElementById("results");
  const spinner = document.getElementById("spinner");
  const output = document.getElementById("output");
  const btn = document.getElementById("classifyBtn");

  results.style.display = "block";
  spinner.style.display = "block";
  spinner.textContent = "Analyzing...";
  output.style.display = "none";
  btn.disabled = true;

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const res = await fetch("/predict", { method: "POST", body: form });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);

    spinner.style.display = "none";
    output.style.display = "block";
  } catch (err) {
    spinner.textContent = "Something went wrong — please try again.";
    console.error(err);
  }

  btn.disabled = false;
}

/**
 * Render prediction results into the DOM.
 *
 * @param {Object} data - Response from /predict endpoint.
 * @param {string} data.prediction - Predicted class label.
 * @param {number} data.confidence - Confidence score 0-1.
 * @param {Array}  data.top5 - Array of {class, confidence} objects.
 * @param {string} data.heatmap_b64 - Base64-encoded GradCAM PNG.
 */
function renderResults(data) {
  // Prediction badge
  document.getElementById("predLabel").textContent = data.prediction;

  // Confidence text
  document.getElementById("confText").textContent =
    `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  // GradCAM heatmap
  document.getElementById("heatmap").src =
    "data:image/png;base64," + data.heatmap_b64;

  // Top 5 probability bars
  const bars = document.getElementById("bars");
  bars.innerHTML = data.top5
    .map(
      (item) => `
      <div class="bar-row">
        <span class="bar-label">${item.class}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${(item.confidence * 100).toFixed(1)}%"></div>
        </div>
        <span class="bar-pct">${(item.confidence * 100).toFixed(1)}%</span>
      </div>`
    )
    .join("");
}
