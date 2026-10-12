# Image2ThreeJS

Drop an image, get an interactive 3D model, entirely in your browser. No server, no uploads.

**Live demo:** https://sumeethedral.github.io/image2threejs/

## How it works

1. The image is downscaled and sent to [Depth Anything V2 (small)](https://huggingface.co/onnx-community/depth-anything-v2-small), running on-device via [Transformers.js](https://github.com/huggingface/transformers.js) (WebGPU, falling back to WASM). The ~25 MB model downloads once and is cached.
2. The depth map displaces a Three.js grid mesh textured with your image.
3. Sliders control depth strength, background cut-off and smoothing. Triangles that span depth cliffs are dropped for a cleaner silhouette.
4. **Download .glb** exports the mesh and texture.

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
