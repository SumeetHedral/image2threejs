import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { WebGPURenderer } from 'three/webgpu';
import { pipeline } from '@huggingface/transformers';

const MODEL_ID = 'onnx-community/depth-anything-v2-small';
const MAX_TEX = 1024; // longest side of the texture / depth input
const STRETCH_LIMIT = 0.3; // drop triangles spanning a depth cliff (normalized depth)
const MAX_SCENE_TRIS = 25e6; // refuse stress settings beyond this
const LOD_DISTANCES = [0, 1.4, 2.2, 3.2]; // multiples of the framing distance
const BENCH_WARMUP_MS = 1000;
const BENCH_SAMPLE_MS = 3000;

const $ = (id) => document.getElementById(id);
const canvasGL = $('view');
const canvasGPU = $('viewGpu');
const stage = $('stage');
const dropEl = $('drop');
const statusEl = $('status');
const controlsEl = $('controls');
const metricsEl = $('metrics');

// ---------- renderers ----------
const glRenderer = new THREE.WebGLRenderer({ canvas: canvasGL, antialias: true });
let gpuRenderer = null; // created lazily on first switch
const scene = new THREE.Scene();
scene.background = new THREE.Color(matchMedia('(prefers-color-scheme: light)').matches ? 0xffffff : 0x181b22);
const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 200);
camera.position.set(0, 0, 3);
const orbit = new OrbitControls(camera, canvasGL);
orbit.enableDamping = true;
orbit.autoRotateSpeed = 1.2;
scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(2, 3, 4);
scene.add(key);
const holder = new THREE.Group();
scene.add(holder);

let active = glRenderer;
let mode = 'webgl';

function resize() {
  const { clientWidth: w, clientHeight: h } = stage;
  for (const r of [glRenderer, gpuRenderer]) {
    if (!r) continue;
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.setSize(w, h, false);
  }
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);
resize();

// ---------- GPU timing ----------
// WebGL: EXT_disjoint_timer_query_webgl2 wrapped around render(). WebGPU: timestamp-query via three.
const gl2 = glRenderer.getContext();
const timerExt = gl2.getExtension('EXT_disjoint_timer_query_webgl2');
const pendingQueries = [];
let glGpuMs = null;
let gpuGpuMs = null;
let resolving = false;

function glTimerBegin() {
  if (!timerExt || pendingQueries.length > 4) return null;
  const q = gl2.createQuery();
  gl2.beginQuery(timerExt.TIME_ELAPSED_EXT, q);
  return q;
}
function glTimerEnd(q) {
  if (!q) return;
  gl2.endQuery(timerExt.TIME_ELAPSED_EXT);
  pendingQueries.push(q);
}
function glTimerPoll() {
  if (!timerExt) return;
  const disjoint = gl2.getParameter(timerExt.GPU_DISJOINT_EXT);
  while (pendingQueries.length && gl2.getQueryParameter(pendingQueries[0], gl2.QUERY_RESULT_AVAILABLE)) {
    const q = pendingQueries.shift();
    if (!disjoint) glGpuMs = gl2.getQueryParameter(q, gl2.QUERY_RESULT) / 1e6;
    gl2.deleteQuery(q);
  }
}
function gpuTimerResolve(renderer) {
  if (resolving) return;
  resolving = true;
  renderer
    .resolveTimestampsAsync('render')
    .then(() => {
      const t = renderer.info.render.timestamp;
      if (t > 0) gpuGpuMs = t;
    })
    .catch(() => {})
    .finally(() => (resolving = false));
}

// ---------- metrics ----------
const win = { start: performance.now(), frames: 0, dt: [], cpu: [] };
let lastNow = 0;
let recording = null; // benchmark sample sink
let lodSummary = '';
let frameDraws = 0; // per-frame draw calls (WebGPU's info.render.calls is cumulative; drawCalls is per frame)

const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const pct = (a, p) => (a.length ? [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(a.length * p))] : 0);
const fmt = (n, d = 1) => (n == null || Number.isNaN(n) ? 'n/a' : n.toFixed(d));
const gpuMsNow = () => (mode === 'webgpu' ? gpuGpuMs : glGpuMs);

function recordFrame(now, cpuMs) {
  if (lastNow) {
    const dt = now - lastNow;
    win.dt.push(dt);
    recording?.dt.push(dt);
  }
  lastNow = now;
  win.cpu.push(cpuMs);
  recording?.cpu.push(cpuMs);
  const g = gpuMsNow();
  if (g != null) recording?.gpu.push(g);
  win.frames++;
  const elapsed = now - win.start;
  if (elapsed >= 500) {
    updateMetricsDom(elapsed);
    win.start = now;
    win.frames = 0;
    win.dt = [];
    win.cpu = [];
  }
}

function backendLabel() {
  if (mode === 'webgpu') {
    const info = gpuRenderer?.backend?.adapter?.info;
    return `WebGPU${info?.description || info?.device ? ` · ${info.description || info.device}` : ''}`;
  }
  const dbg = gl2.getExtension('WEBGL_debug_renderer_info');
  return `WebGL2${dbg ? ` · ${gl2.getParameter(dbg.UNMASKED_RENDERER_WEBGL)}` : ''}`;
}

function updateMetricsDom(elapsed) {
  if (metricsEl.hidden) return;
  const r = active.info.render;
  const gpu = gpuMsNow();
  const fps = (win.frames * 1000) / elapsed;
  metricsEl.textContent =
    `${backendLabel()}\n` +
    `FPS        ${fmt(fps)}   (vsync-capped)\n` +
    `Frame      ${fmt(avg(win.dt), 2)} ms  p95 ${fmt(pct(win.dt, 0.95), 2)}\n` +
    `CPU submit ${fmt(avg(win.cpu), 2)} ms\n` +
    `GPU time   ${gpu == null ? 'n/a (timer query unsupported)' : `${fmt(gpu, 2)} ms`}\n` +
    `Triangles  ${r.triangles.toLocaleString()}   Draws ${frameDraws}\n` +
    `Geometries ${active.info.memory.geometries}   Textures ${active.info.memory.textures}` +
    (lodSummary ? `\nLOD        ${lodSummary}` : '');
}

function summariseLods() {
  if (!lods.length) return (lodSummary = '');
  const counts = {};
  for (const l of lods) counts[l.getCurrentLevel()] = (counts[l.getCurrentLevel()] ?? 0) + 1;
  lodSummary = Object.keys(counts)
    .sort()
    .map((k) => `L${k}×${counts[k]}`)
    .join(' ');
}

function loop(now) {
  orbit.autoRotate = $('spin').checked;
  orbit.update();
  const t0 = performance.now();
  const q = mode === 'webgl' ? glTimerBegin() : null;
  active.render(scene, camera);
  if (mode === 'webgl') {
    glTimerEnd(q);
    glTimerPoll();
  } else {
    gpuTimerResolve(active);
  }
  frameDraws = mode === 'webgpu' ? active.info.render.drawCalls : active.info.render.calls;
  recordFrame(now ?? performance.now(), performance.now() - t0);
}
glRenderer.setAnimationLoop(loop);
setInterval(summariseLods, 500);

// ---------- renderer switching ----------
const modeButtons = { webgl: $('mode-webgl'), webgpu: $('mode-webgpu') };
if (!navigator.gpu) {
  modeButtons.webgpu.disabled = true;
  modeButtons.webgpu.title = 'WebGPU is not available in this browser';
  $('bench').title = 'WebGPU is not available, so there is nothing to compare';
  $('bench').disabled = true;
}

async function setMode(next) {
  if (next === mode) return;
  try {
    if (next === 'webgpu') {
      if (!gpuRenderer) {
        gpuRenderer = new WebGPURenderer({ canvas: canvasGPU, antialias: true, trackTimestamp: true });
        await gpuRenderer.init();
      }
      glRenderer.setAnimationLoop(null);
      active = gpuRenderer;
    } else {
      gpuRenderer?.setAnimationLoop(null);
      active = glRenderer;
    }
    const [on, off] = next === 'webgpu' ? [canvasGPU, canvasGL] : [canvasGL, canvasGPU];
    on.hidden = false;
    off.hidden = true;
    orbit.disconnect();
    orbit.connect(on);
    resize();
    mode = next;
    lastNow = 0;
    win.start = performance.now();
    win.frames = 0;
    win.dt = [];
    win.cpu = [];
    active.setAnimationLoop(loop);
    for (const [k, b] of Object.entries(modeButtons)) b.classList.toggle('on', k === next);
  } catch (err) {
    console.error(err);
    setStatus(`Could not start ${next}: ${err.message}`);
  }
}
modeButtons.webgl.addEventListener('click', () => setMode('webgl'));
modeButtons.webgpu.addEventListener('click', () => setMode('webgpu'));

// ---------- depth model ----------
let depthModel;
function loadModel() {
  depthModel ??= (async () => {
    const progress_callback = (p) => {
      if (p.status === 'progress') setStatus(`Downloading depth model… ${Math.round(p.progress)}% (cached after first visit)`);
    };
    try {
      if (!navigator.gpu) throw new Error('no webgpu');
      return await pipeline('depth-estimation', MODEL_ID, { device: 'webgpu', progress_callback });
    } catch {
      return await pipeline('depth-estimation', MODEL_ID, { device: 'wasm', progress_callback });
    }
  })();
  return depthModel;
}

function setStatus(text) {
  statusEl.hidden = !text;
  statusEl.textContent = text || '';
}

// ---------- image -> mesh ----------
// src: { texCanvas, aspect, depth: Float32Array (normalised 0..1, source resolution), dw, dh, width, height, material }
let src = null;
let levelCache = new Map(); // grid -> { grid, cols, rows, raw, geo, idx }
let lods = [];
let frameDist = 3;

async function handleFile(file) {
  if (!file?.type.startsWith('image/')) return setStatus('That file is not an image.');
  try {
    dropEl.classList.add('hidden');
    setStatus('Reading image…');
    const bmp = await createImageBitmap(file);
    const scale = Math.min(1, MAX_TEX / Math.max(bmp.width, bmp.height));
    const w = Math.round(bmp.width * scale);
    const h = Math.round(bmp.height * scale);
    const tex = document.createElement('canvas');
    tex.width = w;
    tex.height = h;
    tex.getContext('2d').drawImage(bmp, 0, 0, w, h);

    const model = await loadModel();
    setStatus('Estimating depth…');
    const blob = await new Promise((r) => tex.toBlob(r, 'image/png'));
    const url = URL.createObjectURL(blob);
    const { depth } = await model(url);
    URL.revokeObjectURL(url);

    setSource(tex, depth);
    setStatus('');
    controlsEl.hidden = false;
    metricsEl.hidden = !$('showMetrics').checked;
  } catch (err) {
    console.error(err);
    dropEl.classList.remove('hidden');
    setStatus(`Something went wrong: ${err.message}`);
  }
}

function setSource(texCanvas, depthImg) {
  clearScene();
  const { width: dw, height: dh, data } = depthImg;
  const channels = data.length / (dw * dh);
  const d = new Float32Array(dw * dh);
  for (let i = 0; i < d.length; i++) d[i] = data[i * channels];
  // Robust normalisation: 2nd..98th percentile, shared by every LOD level.
  const sorted = Float32Array.from(d).sort();
  const lo = sorted[Math.floor(sorted.length * 0.02)];
  const hi = sorted[Math.floor(sorted.length * 0.98)];
  for (let i = 0; i < d.length; i++) d[i] = THREE.MathUtils.clamp((d[i] - lo) / (hi - lo || 1), 0, 1);

  const aspect = texCanvas.width / texCanvas.height;
  const map = new THREE.CanvasTexture(texCanvas);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = glRenderer.capabilities.getMaxAnisotropy();
  src = {
    texCanvas,
    aspect,
    depth: d,
    dw,
    dh,
    width: aspect >= 1 ? 1.6 : 1.6 * aspect,
    height: aspect >= 1 ? 1.6 / aspect : 1.6,
    material: new THREE.MeshStandardMaterial({ map, side: THREE.DoubleSide, roughness: 0.85, metalness: 0 }),
  };
  rebuildScene();
  frame();
}

function gridSize(grid) {
  const a = src.aspect;
  return {
    cols: a >= 1 ? grid : Math.max(8, Math.round(grid * a)),
    rows: a >= 1 ? Math.max(8, Math.round(grid / a)) : grid,
  };
}

function getLevel(grid) {
  let lv = levelCache.get(grid);
  if (lv) return lv;
  const { cols, rows } = gridSize(grid);
  const { dw, dh, depth, width, height } = src;
  const stride = cols + 1;
  const raw = new Float32Array(stride * (rows + 1));
  for (let j = 0; j <= rows; j++) {
    const fy = (j / rows) * (dh - 1);
    const y0 = Math.floor(fy), y1 = Math.min(dh - 1, y0 + 1), ty = fy - y0;
    for (let i = 0; i <= cols; i++) {
      const fx = (i / cols) * (dw - 1);
      const x0 = Math.floor(fx), x1 = Math.min(dw - 1, x0 + 1), tx = fx - x0;
      raw[j * stride + i] =
        (depth[y0 * dw + x0] * (1 - tx) + depth[y0 * dw + x1] * tx) * (1 - ty) +
        (depth[y1 * dw + x0] * (1 - tx) + depth[y1 * dw + x1] * tx) * ty;
    }
  }
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(raw.length * 3);
  const uv = new Float32Array(raw.length * 2);
  for (let j = 0; j <= rows; j++) {
    for (let i = 0; i <= cols; i++) {
      const k = j * stride + i;
      pos[k * 3] = (i / cols - 0.5) * width;
      pos[k * 3 + 1] = (0.5 - j / rows) * height;
      uv[k * 2] = i / cols;
      uv[k * 2 + 1] = 1 - j / rows;
    }
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setIndex(new THREE.BufferAttribute(new Uint32Array(cols * rows * 6), 1));
  lv = { grid, cols, rows, raw, geo };
  levelCache.set(grid, lv);
  updateLevel(lv);
  return lv;
}

function blur(srcArr, cols, rows, passes) {
  let a = srcArr;
  const stride = cols + 1;
  for (let p = 0; p < passes; p++) {
    const b = new Float32Array(a.length);
    for (let j = 0; j <= rows; j++) {
      for (let i = 0; i <= cols; i++) {
        let sum = 0, n = 0;
        for (let dj = -1; dj <= 1; dj++) {
          const y = j + dj;
          if (y < 0 || y > rows) continue;
          for (let di = -1; di <= 1; di++) {
            const x = i + di;
            if (x < 0 || x > cols) continue;
            sum += a[y * stride + x];
            n++;
          }
        }
        b[j * stride + i] = sum / n;
      }
    }
    a = b;
  }
  return a;
}

function updateLevel(lv) {
  const { cols, rows, raw, geo } = lv;
  const strength = +$('depth').value;
  const cut = +$('cut').value;
  const d = blur(raw, cols, rows, +$('smooth').value);
  const pos = geo.attributes.position;
  for (let k = 0; k < d.length; k++) pos.setZ(k, (d[k] - 0.5) * strength);
  pos.needsUpdate = true;

  // Skip background (below `cut`) and stretched walls across depth cliffs.
  const stride = cols + 1;
  const idx = geo.index.array;
  let n = 0;
  const keep = (a, b, c) => {
    const lo = Math.min(d[a], d[b], d[c]);
    const hi = Math.max(d[a], d[b], d[c]);
    return lo >= cut && hi - lo <= STRETCH_LIMIT;
  };
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const a = j * stride + i, b = a + 1, c = a + stride, e = c + 1;
      if (keep(a, c, b)) { idx[n++] = a; idx[n++] = c; idx[n++] = b; }
      if (keep(b, c, e)) { idx[n++] = b; idx[n++] = c; idx[n++] = e; }
    }
  }
  geo.index.needsUpdate = true;
  geo.setDrawRange(0, n);
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
  lv.triangles = n / 3;
}

function applyParams() {
  if (!src) return;
  for (const lv of levelCache.values()) updateLevel(lv);
  src.material.wireframe = $('wire').checked;
}

// Build the scene objects for the current density / LOD / copies settings.
function rebuildScene() {
  if (!src) return;
  clearObjects();
  const base = +$('density').value;
  const useLod = $('lod').checked;
  let copies = +$('copies').value;

  const { cols, rows } = gridSize(base);
  const maxCopies = Math.max(1, Math.floor(MAX_SCENE_TRIS / (cols * rows * 2)));
  if (copies > maxCopies) {
    const allowed = [100, 25, 9, 1].find((c) => c <= maxCopies) ?? 1;
    setStatus(`${copies} copies at this density would exceed ${MAX_SCENE_TRIS / 1e6}M triangles; using ${allowed}.`);
    setTimeout(() => setStatus(''), 4000);
    copies = allowed;
    $('copies').value = String(allowed);
  }

  const grids = useLod ? [base, base / 2, base / 4, base / 8].map((g) => Math.max(24, Math.round(g))) : [base];
  const levels = grids.map(getLevel);
  frame(); // sets frameDist for LOD thresholds

  const side = Math.ceil(Math.sqrt(copies));
  const gap = Math.max(src.width, src.height) * 1.15;
  for (let n = 0; n < copies; n++) {
    let obj;
    if (useLod) {
      obj = new THREE.LOD();
      levels.forEach((lv, i) => obj.addLevel(new THREE.Mesh(lv.geo, src.material), LOD_DISTANCES[i] * frameDist));
      lods.push(obj);
    } else {
      obj = new THREE.Mesh(levels[0].geo, src.material);
    }
    obj.position.set(((n % side) - (side - 1) / 2) * gap, (((side - 1) / 2) - Math.floor(n / side)) * gap, 0);
    holder.add(obj);
  }
  // Drop cached levels no longer in use.
  for (const [g, lv] of levelCache) {
    if (!grids.includes(g)) {
      lv.geo.dispose();
      levelCache.delete(g);
    }
  }
  src.copies = copies;
  src.side = side;
  src.gap = gap;
  frame();
  applyParams();
}

function frame() {
  if (!src) return;
  const side = src.side ?? 1;
  const w = src.width + (side - 1) * (src.gap ?? 0);
  const h = src.height + (side - 1) * (src.gap ?? 0);
  const fov = THREE.MathUtils.degToRad(camera.fov / 2);
  const single = Math.max(src.width, src.height) / (2 * Math.tan(fov)) * 1.5;
  frameDist = single;
  const dist = Math.max(w / camera.aspect, h) / (2 * Math.tan(fov)) * 1.5;
  camera.position.set(0, 0, Math.max(single, dist));
  orbit.target.set(0, 0, 0);
  orbit.update();
}

function clearObjects() {
  holder.clear();
  lods = [];
}

function clearScene() {
  clearObjects();
  for (const lv of levelCache.values()) lv.geo.dispose();
  levelCache = new Map();
  if (src) {
    src.material.map.dispose();
    src.material.dispose();
    src = null;
  }
}

// ---------- benchmark ----------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function sampleBackend(which) {
  await setMode(which);
  if (mode !== which) return null;
  await sleep(BENCH_WARMUP_MS);
  recording = { dt: [], cpu: [], gpu: [] };
  const trisAt = () => active.info.render;
  await sleep(BENCH_SAMPLE_MS);
  const rec = recording;
  recording = null;
  const r = trisAt();
  return {
    backend: which === 'webgl' ? 'WebGL2' : 'WebGPU',
    frames: rec.dt.length,
    fps: rec.dt.length ? 1000 / avg(rec.dt) : null,
    frameMs: avg(rec.dt),
    p95: pct(rec.dt, 0.95),
    cpu: avg(rec.cpu),
    gpu: rec.gpu.length ? avg(rec.gpu) : null,
    tris: r.triangles,
    draws: frameDraws,
  };
}

async function runBenchmark() {
  if (!src) return;
  const start = mode;
  const btn = $('bench');
  btn.disabled = true;
  const results = [];
  try {
    for (const which of ['webgl', 'webgpu']) {
      setStatus(`Benchmarking ${which === 'webgl' ? 'WebGL2' : 'WebGPU'}… keep this tab visible`);
      const r = await sampleBackend(which);
      if (r) results.push(r);
    }
  } finally {
    await setMode(start);
    setStatus('');
    btn.disabled = false;
  }
  showBenchmark(results);
}

function showBenchmark(results) {
  const el = $('benchResults');
  const best = (key, low = true) => {
    const vals = results.map((r) => r[key]).filter((v) => v != null);
    return vals.length < 2 ? null : low ? Math.min(...vals) : Math.max(...vals);
  };
  const cell = (r, key, d, low = true) => {
    const b = best(key, low);
    const v = r[key];
    return `<td class="${b != null && v === b ? 'win' : ''}">${fmt(v, d)}</td>`;
  };
  const empty = results.some((r) => r.frames === 0);
  const cfg = `${$('density').selectedOptions[0].textContent} · ${$('lod').checked ? 'auto-LOD' : 'fixed'} · ${$('copies').value} cop${$('copies').value === '1' ? 'y' : 'ies'}`;
  el.innerHTML =
    `<h2>Benchmark <small>${cfg}</small></h2>` +
    `<table><thead><tr><th>Backend</th><th>FPS</th><th>Frame ms</th><th>p95 ms</th><th>CPU submit ms</th><th>GPU ms</th><th>Tris</th><th>Draws</th></tr></thead><tbody>` +
    results
      .map(
        (r) =>
          `<tr><th>${r.backend}</th>${cell(r, 'fps', 1, false)}${cell(r, 'frameMs', 2)}${cell(r, 'p95', 2)}${cell(r, 'cpu', 2)}${cell(r, 'gpu', 2)}<td>${r.tris.toLocaleString()}</td><td>${r.draws}</td></tr>`,
      )
      .join('') +
    `</tbody></table>` +
    (empty ? `<p class="warn">No frames were rendered for at least one backend. The browser pauses rendering in hidden tabs, so keep this tab in front and run it again.</p>` : '') +
    `<p>Browsers cap frames at the display refresh rate, so FPS ties until the scene is heavy. Raise Density and Copies, and compare CPU submit and GPU time, which stay meaningful under the cap. GPU time shows n/a when the browser blocks timer queries. Green marks the better value.</p>`;
  el.hidden = false;
}

// ---------- UI wiring ----------
let paramRaf = 0;
function scheduleParams() {
  cancelAnimationFrame(paramRaf);
  paramRaf = requestAnimationFrame(applyParams);
}
for (const id of ['depth', 'cut', 'smooth']) $(id).addEventListener('input', scheduleParams);
$('wire').addEventListener('change', applyParams);
for (const id of ['density', 'lod', 'copies']) $(id).addEventListener('change', () => setTimeout(rebuildScene, 0));
$('showMetrics').addEventListener('change', () => (metricsEl.hidden = !$('showMetrics').checked || !src));
$('bench').addEventListener('click', runBenchmark);

$('reset').addEventListener('click', () => {
  clearScene();
  controlsEl.hidden = true;
  metricsEl.hidden = true;
  $('benchResults').hidden = true;
  dropEl.classList.remove('hidden');
  setStatus('');
});

$('export').addEventListener('click', () => {
  if (!src) return;
  // Export one clean mesh at the chosen density (no spare index slots, no LOD copies).
  const lv = getLevel(+$('density').value);
  const geo = lv.geo.clone();
  geo.setIndex(Array.from(lv.geo.index.array.subarray(0, lv.geo.drawRange.count)));
  geo.clearGroups();
  geo.setDrawRange(0, Infinity);
  new GLTFExporter().parse(
    new THREE.Mesh(geo, src.material),
    (glb) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([glb], { type: 'model/gltf-binary' }));
      a.download = 'image2threejs.glb';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      geo.dispose();
    },
    (err) => setStatus(`Export failed: ${err.message}`),
    { binary: true },
  );
});

$('file').addEventListener('change', (e) => handleFile(e.target.files[0]));
addEventListener('dragover', (e) => {
  e.preventDefault();
  dropEl.classList.add('over');
});
addEventListener('dragleave', () => dropEl.classList.remove('over'));
addEventListener('drop', (e) => {
  e.preventDefault();
  dropEl.classList.remove('over');
  handleFile(e.dataTransfer.files[0]);
});
addEventListener('paste', (e) => {
  const f = [...(e.clipboardData?.files ?? [])].find((x) => x.type.startsWith('image/'));
  if (f) handleFile(f);
});
