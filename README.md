# Image2ThreeJS

Drop an image, get an interactive 3D model, entirely in your browser. No server, no uploads.

**Live demo:** https://sumeethedral.github.io/image2threejs/

## How it works

1. The image is downscaled and sent to [Depth Anything V2 (small)](https://huggingface.co/onnx-community/depth-anything-v2-small), running on-device via [Transformers.js](https://github.com/huggingface/transformers.js) (WebGPU, falling back to WASM). The ~25 MB model downloads once and is cached.
2. The depth map displaces a Three.js grid mesh textured with your image.
3. Sliders control depth strength, background cut-off and smoothing. Triangles that span depth cliffs are dropped for a cleaner silhouette.
4. A **WebGL / WebGPU** switch (top right of the viewer) renders the same scene with either backend. WebGPU is disabled where the browser lacks it.
5. **Metrics and benchmark** (below) let you measure the two backends against each other.
6. **Download .glb** exports the mesh and texture.

## Metrics and LOD

- **Live metrics panel:** backend and GPU name, FPS, frame time (avg and p95), CPU submit time, GPU time, triangles, draw calls and active LOD levels.
- **GPU time:** WebGL uses `EXT_disjoint_timer_query_webgl2` and WebGPU uses timestamp queries. Either shows `n/a` where the browser blocks them.
- **Density:** mesh resolution from 60 up to 960 segments (about 1.4M triangles per copy).
- **Auto-LOD:** uses `THREE.LOD` with four levels (density, /2, /4, /8) switched by camera distance. Zoom out to watch the levels change.
- **Copies:** 1, 9, 25 or 100 instances to raise draw calls. Settings above 25M triangles are refused.
- **Benchmark button:** runs 3 s per backend on the current settings and compares them in a table.

Browsers cap rendering at the display refresh rate, so FPS ties until a scene is heavy. Compare CPU submit and GPU time, which stay meaningful under the cap, and keep the tab visible while benchmarking.

This is a relief reconstruction from a single view: hidden sides are not modelled.

## Develop

```bash
npm install
npm run dev
```

Pushing to `main` deploys to GitHub Pages via `.github/workflows/pages.yml`.

## Roadmap

- Optional "procedural object" mode driven by the [img2threejs](https://github.com/img2threejs/img2threejs) agent skill (needs an LLM backend or user-supplied API key)
- Background removal for true cut-out objects
- Mirror-back and thickness for a closed solid
