# Roadmap

Each stage ends with a working script and a results table, so there is always something to demo.

| Stage | Goal | Main components | Deliverable |
| --- | --- | --- | --- |
| 0 | Foundation | Repo, environment, data format, shared test set | Skeleton, 50-100 test frames |
| 1 | Baseline | RetinaFace (vs. YOLO-face, MediaPipe) + blur / pixelation / solid | Face recall on game frames |
| 2 | Face + inpainting | Face box → SAM mask → LaMa, SD Inpainting | Comparison vs. baseline (LPIPS, time) |
| 3 | Body on images | Mask R-CNN → SAM 2 → LaMa vs. diffusion | Mask IoU on SAIL-VOS, background quality |
| 4 | Body in video | SAM 2 tracking → ProPainter / E2FGVI | Warping error vs. per-frame LaMa |
| 5 | Wrap-up | Own game frames, detector fine-tuning, demo | Report, final presentation |

Optional: on-screen text (nicknames, chat) with PaddleOCR + blur / inpainting.

## Compute

SAM 2, diffusion inpainting and ProPainter need a GPU with roughly 8-12 GB VRAM. Use a local GPU, Google Colab,
or the university server; keep notebooks Colab-friendly.
