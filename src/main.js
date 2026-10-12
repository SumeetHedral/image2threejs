import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { pipeline } from '@huggingface/transformers';

const MODEL_ID = 'onnx-community/depth-anything-v2-small';
const MAX_TEX = 1024; // longest side of the texture / depth input
const GRID = 240; // mesh segments along the longest side
const STRETCH_LIMIT = 0.3; // drop triangles spanning a depth cliff (normalized depth)

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const dropEl = $('drop');
const statusEl = $('status');
const controlsEl = $('controls');

// ---------- three.js scene ----------
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(matchMedia('(prefers-color-scheme: light)').matches ? 0xffffff : 0x181b22);
const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50);
camera.position.set(0, 0, 3);
const orbit = new OrbitControls(camera, canvas);
orbit.enableDamping = true;
orbit.autoRotateSpeed = 1.2;
scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(2, 3, 4);
scene.add(key);

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);
resize();

renderer.setAnimationLoop(() => {
  orbit.autoRotate = $('spin').checked;
  orbit.update();
  renderer.render(scene, camera);
});

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
let current = null; // { mesh, grid, cols, rows, depth (raw normalized), width, height, texture }

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

    buildMesh(tex, depth);
    setStatus('');
    controlsEl.hidden = false;
  } catch (err) {
    console.error(err);
    dropEl.classList.remove('hidden');
    setStatus(`Something went wrong: ${err.message}`);
  }
}

function buildMesh(texCanvas, depthImg) {
  clearMesh();
  const aspect = texCanvas.width / texCanvas.height;
  const cols = aspect >= 1 ? GRID : Math.max(8, Math.round(GRID * aspect));
  const rows = aspect >= 1 ? Math.max(8, Math.round(GRID / aspect)) : GRID;

  // Resample depth onto the grid (bilinear) and normalise to 0..1 using robust percentiles.
  const raw = new Float32Array((cols + 1) * (rows + 1));
  const { width: dw, height: dh, data } = depthImg;
  const channels = data.length / (dw * dh);
  const at = (x, y) => data[(y * dw + x) * channels];
  for (let j = 0; j <= rows; j++) {
    const fy = (j / rows) * (dh - 1);
    const y0 = Math.floor(fy), y1 = Math.min(dh - 1, y0 + 1), ty = fy - y0;
    for (let i = 0; i <= cols; i++) {
      const fx = (i / cols) * (dw - 1);
      const x0 = Math.floor(fx), x1 = Math.min(dw - 1, x0 + 1), tx = fx - x0;
      raw[j * (cols + 1) + i] =
        (at(x0, y0) * (1 - tx) + at(x1, y0) * tx) * (1 - ty) + (at(x0, y1) * (1 - tx) + at(x1, y1) * tx) * ty;
    }
  }
  const sorted = Float32Array.from(raw).sort();
  const lo = sorted[Math.floor(sorted.length * 0.02)];
  const hi = sorted[Math.floor(sorted.length * 0.98)];
  for (let i = 0; i < raw.length; i++) raw[i] = THREE.MathUtils.clamp((raw[i] - lo) / (hi - lo || 1), 0, 1);

  const width = aspect >= 1 ? 1.6 : 1.6 * aspect;
  const height = aspect >= 1 ? 1.6 / aspect : 1.6;

  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(raw.length * 3);
  const uv = new Float32Array(raw.length * 2);
  for (let j = 0; j <= rows; j++) {
    for (let i = 0; i <= cols; i++) {
      const k = j * (cols + 1) + i;
      pos[k * 3] = (i / cols - 0.5) * width;
      pos[k * 3 + 1] = (0.5 - j / rows) * height;
      uv[k * 2] = i / cols;
      uv[k * 2 + 1] = 1 - j / rows;
    }
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));

  const map = new THREE.CanvasTexture(texCanvas);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const mat = new THREE.MeshStandardMaterial({ map, side: THREE.DoubleSide, roughness: 0.85, metalness: 0 });
  const mesh = new THREE.Mesh(geo, mat);
  scene.add(mesh);

  current = { mesh, raw, cols, rows, width, height };
  applyParams();
  frame(width, height);
}

function blur(src, cols, rows, passes) {
  let a = src;
  for (let p = 0; p < passes; p++) {
    const b = new Float32Array(a.length);
    const stride = cols + 1;
    for (let j = 0; j <= rows; j++) {
      for (let i = 0; i <= cols; i++) {
        let sum = 0, n = 0;
        for (let dj = -1; dj <= 1; dj++) {
          for (let di = -1; di <= 1; di++) {
            const x = i + di, y = j + dj;
            if (x < 0 || y < 0 || x > cols || y > rows) continue;
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

function applyParams() {
  if (!current) return;
  const { mesh, raw, cols, rows } = current;
  const strength = +$('depth').value;
  const cut = +$('cut').value;
  const d = blur(raw, cols, rows, +$('smooth').value);
  const pos = mesh.geometry.attributes.position;
  for (let k = 0; k < d.length; k++) pos.setZ(k, (d[k] - 0.5) * strength);
  pos.needsUpdate = true;

  // Rebuild triangles: skip background (below `cut`) and stretched walls across depth cliffs.
  const stride = cols + 1;
  const idx = [];
  const keep = (a, b, c) => {
    const lo = Math.min(d[a], d[b], d[c]);
    const hi = Math.max(d[a], d[b], d[c]);
    return lo >= cut && hi - lo <= STRETCH_LIMIT;
  };
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const a = j * stride + i, b = a + 1, c = a + stride, e = c + 1;
      if (keep(a, c, b)) idx.push(a, c, b);
      if (keep(b, c, e)) idx.push(b, c, e);
    }
  }
  mesh.geometry.setIndex(idx);
  mesh.geometry.computeVertexNormals();
  mesh.geometry.computeBoundingSphere();
  mesh.material.wireframe = $('wire').checked;
}

function frame(w, h) {
  const dist = Math.max(w, h) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.5;
  camera.position.set(0, 0, dist);
  orbit.target.set(0, 0, 0);
  orbit.update();
}

function clearMesh() {
  if (!current) return;
  scene.remove(current.mesh);
  current.mesh.geometry.dispose();
  current.mesh.material.map.dispose();
  current.mesh.material.dispose();
  current = null;
}

// ---------- UI wiring ----------
for (const id of ['depth', 'cut', 'smooth']) $(id).addEventListener('input', applyParams);
$('wire').addEventListener('change', applyParams);

$('reset').addEventListener('click', () => {
  clearMesh();
  controlsEl.hidden = true;
  dropEl.classList.remove('hidden');
  setStatus('');
});

$('export').addEventListener('click', () => {
  if (!current) return;
  new GLTFExporter().parse(
    current.mesh,
    (glb) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([glb], { type: 'model/gltf-binary' }));
      a.download = 'image2threejs.glb';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
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
