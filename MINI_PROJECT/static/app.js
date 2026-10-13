/**
 * Neural MNIST Client Application
 * Smooth canvas drawing, real-time prediction debouncing, and dynamic visualizers.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const canvas = document.getElementById('drawing-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const canvasHint = document.getElementById('canvas-hint');
  const canvasFrame = document.querySelector('.canvas-frame');
  const brushSlider = document.getElementById('brush-size');
  const brushVal = document.getElementById('brush-val');
  const btnEraser = document.getElementById('btn-eraser');
  const btnClear = document.getElementById('btn-clear');
  const btnSample = document.getElementById('btn-sample');

  const predictedDigitEl = document.getElementById('predicted-digit');
  const predictedConfEl = document.getElementById('predicted-confidence');
  const latencyValEl = document.getElementById('latency-val');
  const headerLatencyEl = document.getElementById('header-latency');
  const headerAccuracyEl = document.getElementById('header-accuracy');
  const tensorPreviewImg = document.getElementById('tensor-preview-img');
  const probListEl = document.getElementById('probability-list');

  // Drawing state
  let isDrawing = false;
  let isEraser = false;
  let hasStrokes = false;
  let lastX = 0;
  let lastY = 0;
  let strokePoints = [];
  let brushSize = parseInt(brushSlider.value, 10) || 24;

  // Real-time prediction throttling
  let predictTimeout = null;
  let isPredicting = false;
  let pendingPredict = false;

  // Initialize Probability Bars (0-9)
  function initProbabilityBars() {
    probListEl.innerHTML = '';
    for (let i = 0; i < 10; i++) {
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.id = `prob-row-${i}`;
      row.innerHTML = `
        <span class="prob-digit">${i}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" id="prob-fill-${i}"></div>
        </div>
        <span class="prob-value" id="prob-val-${i}">0.0%</span>
      `;
      probListEl.appendChild(row);
    }
  }

  // Setup Canvas Dimensions with High DPI support
  function setupCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Initial background: Solid dark
    clearCanvas(false);
  }

  function clearCanvas(triggerPrediction = true) {
    const rect = canvas.getBoundingClientRect();
    ctx.fillStyle = '#030712';
    ctx.fillRect(0, 0, rect.width, rect.height);
    hasStrokes = false;
    canvasHint.classList.remove('hidden');

    if (triggerPrediction) {
      resetPredictionUI();
      // Send an empty predict request to refresh tensor view
      schedulePrediction(0);
    }
  }

  function resetPredictionUI() {
    predictedDigitEl.textContent = '—';
    predictedDigitEl.classList.remove('pop');
    predictedConfEl.textContent = '0.0%';
    latencyValEl.textContent = '—';
    tensorPreviewImg.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='112' height='112'%3E%3Crect width='112' height='112' fill='%230b0f19'/%3E%3C/svg%3E";

    for (let i = 0; i < 10; i++) {
      const row = document.getElementById(`prob-row-${i}`);
      const fill = document.getElementById(`prob-fill-${i}`);
      const val = document.getElementById(`prob-val-${i}`);
      if (row) row.classList.remove('active');
      if (fill) fill.style.width = '0%';
      if (val) val.textContent = '0.0%';
    }
  }

  // Canvas Coordinates Helper
  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  }

  // Drawing Events
  function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    canvasFrame.classList.add('drawing-active');
    canvasHint.classList.add('hidden');
    hasStrokes = true;

    const pos = getPos(e);
    lastX = pos.x;
    lastY = pos.y;
    strokePoints = [pos];

    // Single dot on click
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = isEraser ? '#030712' : '#ffffff';
    ctx.fill();

    schedulePrediction(80);
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const pos = getPos(e);
    strokePoints.push(pos);

    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = isEraser ? '#030712' : '#ffffff';

    if (strokePoints.length > 2) {
      // Quadratic Bezier curve smoothing
      const p1 = strokePoints[strokePoints.length - 2];
      const p2 = strokePoints[strokePoints.length - 1];
      const midPoint = {
        x: (p1.x + p2.x) / 2,
        y: (p1.y + p2.y) / 2
      };

      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.quadraticCurveTo(p1.x, p1.y, midPoint.x, midPoint.y);
      ctx.stroke();

      lastX = midPoint.x;
      lastY = midPoint.y;
    } else {
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();
      lastX = pos.x;
      lastY = pos.y;
    }

    schedulePrediction(70);
  }

  function stopDrawing(e) {
    if (!isDrawing) return;
    if (e) e.preventDefault();
    isDrawing = false;
    canvasFrame.classList.remove('drawing-active');
    strokePoints = [];

    // Trigger immediate prediction when user lifts pen/mouse
    schedulePrediction(0);
  }

  // Real-time prediction scheduling
  function schedulePrediction(delayMs = 60) {
    if (predictTimeout) {
      clearTimeout(predictTimeout);
    }
    predictTimeout = setTimeout(() => {
      sendPredictRequest();
    }, delayMs);
  }

  async function sendPredictRequest() {
    if (isPredicting) {
      pendingPredict = true;
      return;
    }

    isPredicting = true;
    const dataUrl = canvas.toDataURL('image/png');

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const result = await response.json();
      updatePredictionUI(result);
    } catch (err) {
      console.warn('Prediction request failed:', err);
    } finally {
      isPredicting = false;
      if (pendingPredict) {
        pendingPredict = false;
        schedulePrediction(40);
      }
    }
  }

  function updatePredictionUI(result) {
    if (result.empty) {
      resetPredictionUI();
      return;
    }

    // Top prediction
    const prevDigit = predictedDigitEl.textContent;
    predictedDigitEl.textContent = result.prediction;
    if (prevDigit !== String(result.prediction)) {
      predictedDigitEl.classList.remove('pop');
      void predictedDigitEl.offsetWidth; // Trigger reflow for CSS animation
      predictedDigitEl.classList.add('pop');
    }

    // Confidence
    const confPercent = (result.confidence * 100).toFixed(1);
    predictedConfEl.textContent = `${confPercent}%`;

    // Latency
    const latStr = `${result.latency_ms.toFixed(0)}ms`;
    latencyValEl.textContent = latStr;
    headerLatencyEl.textContent = latStr;

    // Tensor Preview Image
    if (result.preview) {
      tensorPreviewImg.src = result.preview;
    }

    // Update 10 Probability Bars
    const probs = result.probabilities || [];
    for (let i = 0; i < 10; i++) {
      const p = probs[i] !== undefined ? probs[i] : 0;
      const pct = (p * 100).toFixed(1);
      const row = document.getElementById(`prob-row-${i}`);
      const fill = document.getElementById(`prob-fill-${i}`);
      const val = document.getElementById(`prob-val-${i}`);

      if (fill) fill.style.width = `${pct}%`;
      if (val) val.textContent = `${pct}%`;

      if (row) {
        if (i === result.prediction) {
          row.classList.add('active');
        } else {
          row.classList.remove('active');
        }
      }
    }
  }

  // Load Random MNIST Sample
  async function loadRandomSample() {
    try {
      btnSample.disabled = true;
      btnSample.style.opacity = '0.6';

      const res = await fetch('/api/sample');
      if (!res.ok) throw new Error('Sample fetch failed');
      const data = await res.json();

      const img = new Image();
      img.onload = () => {
        clearCanvas(false);
        const rect = canvas.getBoundingClientRect();
        // Center the 28x28 sample nicely on our 300x300 canvas
        const sampleSize = 220;
        const xOffset = (rect.width - sampleSize) / 2;
        const yOffset = (rect.height - sampleSize) / 2;

        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(img, xOffset, yOffset, sampleSize, sampleSize);
        hasStrokes = true;
        canvasHint.classList.add('hidden');

        // Immediately trigger prediction
        schedulePrediction(0);
        btnSample.disabled = false;
        btnSample.style.opacity = '1';
      };
      img.src = data.image;
    } catch (e) {
      console.error('Failed to load sample:', e);
      btnSample.disabled = false;
      btnSample.style.opacity = '1';
    }
  }

  // Fetch Model Info on Startup
  async function loadModelInfo() {
    try {
      const res = await fetch('/api/model-info');
      if (res.ok) {
        const info = await res.json();
        if (info.test_accuracy_percentage) {
          headerAccuracyEl.textContent = info.test_accuracy_percentage;
        }
      }
    } catch (e) {
      console.warn('Could not load model info:', e);
    }
  }

  // Event Listeners: Canvas
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  window.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseleave', () => {
    if (isDrawing) stopDrawing();
  });

  // Touch Support
  canvas.addEventListener('touchstart', startDrawing, { passive: false });
  canvas.addEventListener('touchmove', draw, { passive: false });
  window.addEventListener('touchend', stopDrawing, { passive: false });
  window.addEventListener('touchcancel', stopDrawing, { passive: false });

  // Brush Slider
  brushSlider.addEventListener('input', (e) => {
    brushSize = parseInt(e.target.value, 10);
    brushVal.textContent = `${brushSize}px`;
  });

  // Eraser Toggle
  btnEraser.addEventListener('click', () => {
    isEraser = !isEraser;
    btnEraser.classList.toggle('active', isEraser);
  });

  // Clear Canvas
  btnClear.addEventListener('click', () => {
    clearCanvas(true);
  });

  // Sample Button
  btnSample.addEventListener('click', loadRandomSample);

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'c' || e.key === 'C' || e.key === 'Escape') {
      clearCanvas(true);
    } else if (e.key === 's' || e.key === 'S') {
      loadRandomSample();
    }
  });

  // Initialization
  initProbabilityBars();
  setupCanvas();
  loadModelInfo();
  window.addEventListener('resize', setupCanvas);
});
