const input = document.getElementById('file-input');
const zone = document.getElementById('drop-zone');
const browse = document.getElementById('browse');
const analyze = document.getElementById('analyze');
const preview = document.getElementById('preview');
const empty = document.getElementById('upload-empty');
const removeImage = document.getElementById('remove-image');
const notice = document.getElementById('notice');
const result = document.getElementById('result');
const resultPlaceholder = document.getElementById('result-placeholder');
const resultLoading = document.getElementById('result-loading');
const resultState = document.getElementById('result-state');
const paletteButton = document.getElementById('palette-toggle');
const paletteName = document.getElementById('palette-name');
const historyKey = 'hemalens-score-history';
const history = loadHistory();
const palettes = ['aqua', 'blue', 'violet'];
const paletteLabels = { aqua: 'Aqua', blue: 'Blue', violet: 'Violet' };
let selected = null;
let previewUrl = null;

function setPalette(palette, persist = false) {
  const activePalette = palettes.includes(palette) ? palette : palettes[0];
  document.body.dataset.palette = activePalette;
  paletteName.textContent = paletteLabels[activePalette];
  paletteButton.setAttribute(
    'aria-label',
    `Change color palette. Current palette: ${paletteLabels[activePalette]}`,
  );
  document.querySelector('meta[name="theme-color"]').content =
    activePalette === 'blue' ? '#10182b' : activePalette === 'violet' ? '#1b1427' : '#08151b';
  if (persist) {
    try {
      localStorage.setItem('malaria-detection-palette', activePalette);
    } catch (error) {
      console.warn('Could not save the selected color palette.', error);
    }
  }
}

try {
  setPalette(localStorage.getItem('malaria-detection-palette') || palettes[0]);
} catch (error) {
  console.warn('Could not restore the selected color palette.', error);
  setPalette(palettes[0]);
}

paletteButton.addEventListener('click', () => {
  const currentIndex = palettes.indexOf(document.body.dataset.palette);
  setPalette(palettes[(currentIndex + 1) % palettes.length], true);
});

browse.addEventListener('click', event => {
  event.stopPropagation();
  input.click();
});

zone.addEventListener('click', event => {
  if (!event.target.closest('button')) input.click();
});

input.addEventListener('change', () => setFile(input.files[0]));

['dragenter', 'dragover'].forEach(name => {
  zone.addEventListener(name, event => {
    event.preventDefault();
    zone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(name => {
  zone.addEventListener(name, event => {
    event.preventDefault();
    zone.classList.remove('dragover');
  });
});

zone.addEventListener('drop', event => setFile(event.dataTransfer.files[0]));

function setFile(file) {
  notice.hidden = true;
  if (!file) return;
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) {
    showError('Choose a PNG, JPG, or WebP image.');
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    showError('Image exceeds the 12 MB limit.');
    return;
  }

  selected = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  empty.hidden = true;
  removeImage.hidden = false;
  document.getElementById('specimen-meta').textContent =
    `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  analyze.disabled = false;
  resultPlaceholder.hidden = false;
  resultLoading.hidden = true;
  result.hidden = true;
  resultState.classList.remove('active');
  resultState.innerHTML = '<i></i> READY TO ANALYZE';
}

function loadHistory() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(historyKey) || '[]');
    if (!Array.isArray(saved)) return [];
    return saved
      .filter(entry =>
        entry &&
        Number.isFinite(entry.confidence) &&
        entry.confidence >= 0 &&
        entry.confidence <= 100 &&
        (entry.prediction === 'Parasitized' || entry.prediction === 'Uninfected') &&
        (typeof entry.timestamp === 'string' || entry.timestamp === undefined),
      )
      .map(entry => ({
        confidence: entry.confidence,
        prediction: entry.prediction,
        timestamp: entry.timestamp,
        parasitizedScore:
          Number.isFinite(entry.parasitizedScore) &&
          entry.parasitizedScore >= 0 &&
          entry.parasitizedScore <= 100
            ? entry.parasitizedScore
            : entry.prediction === 'Parasitized'
              ? entry.confidence
              : 100 - entry.confidence,
      }))
      .slice(-10);
  } catch (error) {
    console.warn('Could not restore this tab’s score history.', error);
    return [];
  }
}

removeImage.addEventListener('click', () => {
  selected = null;
  input.value = '';
  preview.hidden = true;
  preview.removeAttribute('src');
  empty.hidden = false;
  removeImage.hidden = true;
  analyze.disabled = true;
  document.getElementById('specimen-meta').textContent = 'NO IMAGE SELECTED';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  result.hidden = true;
  resultLoading.hidden = true;
  resultPlaceholder.hidden = false;
  resultState.classList.remove('active');
  resultState.innerHTML = '<i></i> AWAITING IMAGE';
});

function showError(message) {
  notice.textContent = message;
  notice.hidden = false;
}

const svgNamespace = 'http://www.w3.org/2000/svg';

function svgNode(name, attributes = {}, text = null) {
  const node = document.createElementNS(svgNamespace, name);
  Object.entries(attributes).forEach(([key, value]) => {
    node.setAttribute(key, String(value));
  });
  if (text !== null) node.textContent = text;
  return node;
}

function addChartFrame(chart, titleText, descriptionText, maxValue, ticks, suffix = '') {
  chart.replaceChildren();
  chart.append(svgNode('title', {}, titleText), svgNode('desc', {}, descriptionText));
  const left = 42;
  const right = 350;
  const top = 20;
  const bottom = 132;
  ticks.forEach(value => {
    const y = bottom - (value / maxValue) * (bottom - top);
    chart.append(
      svgNode('line', {
        x1: left,
        x2: right,
        y1: y,
        y2: y,
        class: 'chart-gridline',
      }),
      svgNode(
        'text',
        { x: left - 7, y: y + 3, class: 'chart-axis-label', 'text-anchor': 'end' },
        `${value}${suffix}`,
      ),
    );
  });
  return { left, right, top, bottom };
}

function renderHistogram() {
  const chart = document.getElementById('load-histogram');
  const scores = history.map(entry => entry.parasitizedScore);
  const bins = Array.from({ length: 10 }, () => 0);
  scores.forEach(score => {
    const index = Math.min(9, Math.floor(score / 10));
    bins[index] += 1;
  });
  document.getElementById('histogram-count').textContent = String(scores.length);
  const maxCount = Math.max(1, ...bins);
  const ticks = [...new Set([0, Math.ceil(maxCount / 2), maxCount])];
  const frame = addChartFrame(
    chart,
    'Parasitized model-score histogram',
    `Histogram of ${scores.length} image-level parasitized scores in ten-percent score bins. These are classifier scores, not infected-cell counts.`,
    maxCount,
    ticks,
  );
  const defs = svgNode('defs');
  const histogramGradient = svgNode('linearGradient', {
    id: 'histogram-fill-gradient',
    x1: 0,
    x2: 0,
    y1: 0,
    y2: 1,
  });
  histogramGradient.append(
    svgNode('stop', { offset: '0%', 'stop-color': '#93e3c1', 'stop-opacity': 0.9 }),
    svgNode('stop', { offset: '100%', 'stop-color': '#43bfae', 'stop-opacity': 0.42 }),
  );
  defs.append(histogramGradient);
  chart.append(defs);
  const plotWidth = frame.right - frame.left;
  const slot = plotWidth / bins.length;
  const chartHeight = frame.bottom - frame.top;

  bins.forEach((count, index) => {
    const height = (count / maxCount) * chartHeight;
    const x = frame.left + index * slot + 4;
    const y = frame.bottom - height;
    const bar = svgNode('rect', {
      x,
      y,
      width: slot - 8,
      height: Math.max(0, height),
      rx: 3,
      class: 'histogram-bar',
    });
    bar.append(
      svgNode(
        'title',
        {},
        `${index * 10}–${(index + 1) * 10}% parasitized score: ${count} image analyses`,
      ),
    );
    chart.append(bar);
    if (count > 0) {
      chart.append(
        svgNode(
          'text',
          {
            x: x + (slot - 8) / 2,
            y: Math.max(frame.top + 9, y - 5),
            class: 'chart-value-label',
            'text-anchor': 'middle',
          },
          String(count),
        ),
      );
    }
  });

  [0, 20, 40, 60, 80, 100].forEach(value => {
    const x = frame.left + (value / 100) * plotWidth;
    chart.append(
      svgNode(
        'text',
        {
          x,
          y: 153,
          class: 'chart-axis-label',
          'text-anchor': value === 0 ? 'start' : value === 100 ? 'end' : 'middle',
        },
        String(value),
      ),
    );
  });
  chart.append(
    svgNode(
      'text',
      {
        x: 10,
        y: (frame.top + frame.bottom) / 2,
        class: 'chart-axis-title histogram-y-title',
        'text-anchor': 'middle',
        transform: `rotate(-90 10 ${(frame.top + frame.bottom) / 2})`,
      },
      'IMAGE COUNT',
    ),
    svgNode(
      'text',
      { x: 196, y: 168, class: 'chart-axis-title', 'text-anchor': 'middle' },
      'PARASITIZED MODEL SCORE (%)',
    ),
  );
}

function renderPerformanceMetrics(metrics) {
  const chart = document.getElementById('performance-chart');
  const accuracy = document.getElementById('validation-accuracy');
  const auc = document.getElementById('validation-auc');
  const note = document.getElementById('metrics-note');
  if (!metrics.available) {
    accuracy.textContent = '—';
    auc.textContent = '—';
    chart.replaceChildren(
      svgNode('title', {}, 'Validation metrics unavailable'),
      svgNode('desc', {}, metrics.error),
    );
    note.textContent = metrics.error;
    return;
  }

  accuracy.textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
  auc.textContent = metrics.roc_auc.toFixed(3);
  note.textContent =
    `Validation set: ${metrics.validation_count.toLocaleString()} images. ${metrics.split_note}`;
  const frame = addChartFrame(
    chart,
    'Grouped validation metrics',
    `Validation precision, recall and F1 score for the Parasitized and Uninfected classes. Accuracy ${accuracy.textContent}; ROC AUC ${auc.textContent}.`,
    100,
    [0, 50, 100],
    '%',
  );
  const groups = [
    { key: 'precision', label: 'Precision' },
    { key: 'recall', label: 'Recall' },
    { key: 'f1-score', label: 'F1 score' },
  ];
  const classSeries = [
    { name: 'Parasitized', className: 'performance-infected' },
    { name: 'Uninfected', className: 'performance-healthy' },
  ];
  const groupSlot = (frame.right - frame.left) / groups.length;
  const barWidth = 22;
  const chartHeight = frame.bottom - frame.top;

  groups.forEach((group, groupIndex) => {
    const center = frame.left + (groupIndex + 0.5) * groupSlot;
    classSeries.forEach((series, seriesIndex) => {
      const score = metrics.classes[series.name][group.key] * 100;
      const height = (score / 100) * chartHeight;
      const x = center + (seriesIndex === 0 ? -barWidth - 2 : 2);
      const bar = svgNode('rect', {
        x,
        y: frame.bottom - height,
        width: barWidth,
        height,
        rx: 3,
        class: `performance-bar ${series.className}`,
      });
      bar.append(svgNode('title', {}, `${series.name} ${group.label}: ${score.toFixed(1)}%`));
      chart.append(bar);
    });
    chart.append(
      svgNode(
        'text',
        { x: center, y: 153, class: 'chart-axis-label', 'text-anchor': 'middle' },
        group.label,
      ),
    );
  });
}

async function loadPerformanceMetrics() {
  try {
    const response = await fetch('/metrics');
    const metrics = await response.json().catch(() => ({}));
    if (!response.ok || !metrics.available) {
      throw new Error(metrics.error || `Validation metrics unavailable (${response.status}).`);
    }
    renderPerformanceMetrics(metrics);
  } catch (error) {
    renderPerformanceMetrics({ available: false, error: error.message });
  }
}

function renderTrend() {
  const chart = document.getElementById('trend-chart');
  const points = history.slice(-10);
  const currentScore = points.length
    ? points[points.length - 1].parasitizedScore
    : null;
  document.getElementById('trend-current').textContent =
    currentScore === null ? '—' : `${currentScore.toFixed(1)}%`;
  const frame = addChartFrame(
    chart,
    'Historical parasitized model-score trend',
    `Showing ${points.length} of up to 10 recent image analyses. Vertical scale is 0 to 100 percent. Scores are not regional or patient history.`,
    100,
    [0, 50, 100],
    '%',
  );
  const defs = svgNode('defs');
  const fillGradient = svgNode('linearGradient', {
    id: 'trend-fill-gradient',
    x1: 0,
    x2: 0,
    y1: 0,
    y2: 1,
  });
  fillGradient.append(
    svgNode('stop', { offset: '0%', 'stop-color': '#93e3c1', 'stop-opacity': 0.2 }),
    svgNode('stop', { offset: '100%', 'stop-color': '#93e3c1', 'stop-opacity': 0 }),
  );
  defs.append(fillGradient);
  chart.append(defs);
  if (!points.length) return;

  const plotted = points.map((entry, index) => ({
    x:
      points.length === 1
        ? (frame.left + frame.right) / 2
        : frame.left + index * ((frame.right - frame.left) / (points.length - 1)),
    y: frame.bottom - (entry.parasitizedScore / 100) * (frame.bottom - frame.top),
    entry,
  }));
  if (plotted.length > 1) {
    const fill = svgNode('polygon', {
      points: `${plotted[0].x},${frame.bottom} ${plotted.map(point => `${point.x},${point.y}`).join(' ')} ${plotted[plotted.length - 1].x},${frame.bottom}`,
      class: 'trend-area',
    });
    chart.append(fill);
    chart.append(
      svgNode('polyline', {
        points: plotted.map(point => `${point.x},${point.y}`).join(' '),
        class: 'trend-line',
      }),
    );
  }
  plotted.forEach((point, index) => {
    const circle = svgNode('circle', {
      cx: point.x,
      cy: point.y,
      r: index === plotted.length - 1 ? 4.5 : 3.5,
      class: index === plotted.length - 1 ? 'trend-point latest' : 'trend-point',
    });
    const time = point.entry.timestamp
      ? new Date(point.entry.timestamp).toLocaleString()
      : 'Earlier analysis';
    circle.append(
      svgNode(
        'title',
        {},
        `${point.entry.prediction}: ${point.entry.parasitizedScore.toFixed(2)}% parasitized score · ${time}`,
      ),
    );
    chart.append(circle);
  });
}

document.getElementById('clear-history').addEventListener('click', () => {
  history.length = 0;
  try {
    sessionStorage.removeItem(historyKey);
  } catch (error) {
    console.warn('Could not clear this tab’s saved score history.', error);
  }
  renderHistogram();
  renderTrend();
});

renderHistogram();
renderTrend();
loadPerformanceMetrics();

analyze.addEventListener('click', async () => {
  if (!selected) return;
  analyze.disabled = true;
  analyze.querySelector('span:first-child').textContent = 'Analyzing image…';
  notice.hidden = true;
  resultPlaceholder.hidden = true;
  result.hidden = true;
  resultLoading.hidden = false;
  resultState.classList.remove('active');
  resultState.innerHTML = '<i></i> ANALYZING';

  try {
    const formData = new FormData();
    formData.append('file', selected);
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.success) {
      throw new Error(data.error || `Prediction request failed (${response.status}).`);
    }

    const parasitized = data.probabilities.Parasitized;
    const uninfected = data.probabilities.Uninfected;
    document.getElementById('confidence').textContent = `${data.confidence.toFixed(1)}%`;
    document.getElementById('confidence-donut').style.setProperty(
      '--parasitized-angle',
      `${parasitized}%`,
    );
    document.getElementById('prediction').textContent = data.prediction;
    document.getElementById('prediction').classList.toggle(
      'infected-prediction',
      data.prediction === 'Parasitized',
    );
    document.getElementById('p-infected').textContent = `${parasitized.toFixed(1)}%`;
    document.getElementById('p-healthy').textContent = `${uninfected.toFixed(1)}%`;

    document.getElementById('timestamp').textContent =
      `ANALYZED · ${new Date(data.timestamp).toLocaleString()}`;
    history.push({
      confidence: data.confidence,
      prediction: data.prediction,
      timestamp: data.timestamp,
      parasitizedScore: parasitized,
    });
    if (history.length > 10) history.shift();
    try {
      sessionStorage.setItem(historyKey, JSON.stringify(history));
    } catch (error) {
      console.warn('Could not save this tab’s score history.', error);
    }
    renderHistogram();
    renderTrend();

    resultLoading.hidden = true;
    result.hidden = false;
    resultState.classList.add('active');
    resultState.innerHTML = '<i></i> ANALYSIS COMPLETE';
  } catch (error) {
    resultLoading.hidden = true;
    resultPlaceholder.hidden = false;
    resultState.innerHTML = '<i></i> ANALYSIS FAILED';
    showError(error.message || 'Prediction request failed.');
  } finally {
    analyze.disabled = false;
    analyze.querySelector('span:first-child').textContent = 'Analyze image';
  }
});
