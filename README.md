# Identity-Aware Video Object Removal Pipeline

Research prototype for user-guided video object removal with a focus on
crowded scenes, target identity preservation, VOID-compatible quadmask export,
and reproducible diagnostics.

This project is not a new video inpainting foundation model. The contribution
is the pipeline around inpainting:

1. Select the target object or person from a reference frame.
2. Generate full-video masks with static, SAM2/SAMURAI-style, or crowded-human
   tracking modes.
3. Preserve target identity in crowded human scenes with YOLO, BoT-SORT/ReID,
   and SAM2 silhouette refinement.
4. Build VOID-compatible quadmasks with primary, affected, overlap, and
   preserved regions.
5. Split long videos into resource-aware VOID chunks.
6. Run or export Netflix VOID Pass 1 packages.
7. Retime, validate, stitch, and optionally restore the output video.

Sample input/output videos are available here:

https://drive.google.com/drive/folders/1_F3iFmOpJdu-ifwlMnnVD-FrFMmYoXOl?usp=sharing

## Why This Exists

Modern video object removal models can produce strong results on simple scenes,
but real clips fail in ways that are hard to debug:

- the mask follows the wrong person,
- neighboring people leak into the target mask,
- shadows or floor contact are not marked as affected regions,
- low-resolution model output becomes blurry after naive upscaling,
- chunked model output has wrong frame timing,
- final videos flicker after frame-only restoration.

This repository treats those problems as separate failure surfaces. The goal is
to make each stage inspectable before spending GPU time on VOID inference.

## Architecture

```text
Input video
  -> user ROI selection
  -> mask generation / identity tracking
  -> affected-region reasoning
  -> VOID quadmask export
  -> resource-aware chunk packages
  -> optional VOID Pass 1 execution
  -> chunk retiming and stitching
  -> optional quality restoration
  -> final video
```

For crowded-human removal, the intended mask path is:

```text
YOLO person detection
  -> BoT-SORT/ReID identity tracking
  -> SAM2 image-prompt silhouette refinement
  -> neighbor-person exclusion
  -> full-video mask cache
  -> VOID quadmask export
```

## Repository Layout

```text
main.py                         FastAPI web backend
web/                            browser UI
src/logo_removal/               pipeline package
scripts/                        research/export/diagnostic scripts
tests/                          focused unit tests
docs/                           research notes and article drafts
in_colab.ipynb                   Colab VOID Pass 1 notebook
run_main_fastapi.ipynb           Colab/FastAPI launch notebook
sample_videos/                  local demo videos, ignored by default
```

Core files:

```text
src/logo_removal/void_pipeline.py   full-video VOID orchestration
src/logo_removal/void_export.py     VOID package and quadmask export
src/logo_removal/vlm_analysis.py    heuristic/Ollama affected-region logic
src/logo_removal/mask_providers.py  static, SAM2, SAMURAI, and crowded-human masks
src/logo_removal/tasks.py           Celery job execution
src/logo_removal/pipeline.py        legacy local OpenCV/LaMa processing path
```

## Requirements

Minimum local requirements:

- Python 3.10+
- FFmpeg and FFprobe on `PATH`
- Redis server for the browser UI
- Python packages from `requirements.txt`

Install:

```bash
python3 -m pip install -r requirements.txt
```

Check FFmpeg:

```bash
ffmpeg -version
ffprobe -version
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Some tests and modes are dependency-gated. SAM2, SAMURAI, DAM4SAM, BoxMOT,
Ultralytics, VOID, VEnhancer, and Real-ESRGAN require separate model/repo setup.

## Browser UI

Start Redis:

```bash
redis-server
```

Start a Celery worker:

```bash
celery -A src.logo_removal.tasks.celery_app worker --loglevel=INFO --pool=solo --concurrency=1
```

Start FastAPI:

```bash
python3 main.py
```

Then open:

```text
http://127.0.0.1:8000
```

The UI supports:

- video upload,
- target ROI selection,
- legacy fast inpainting mode,
- SAM2 + VOID package generation,
- crowded-human removal mode,
- VOID resource profiles,
- affected-region settings,
- progress logs,
- output download.

## CLI Examples

Legacy local OpenCV fallback:

```bash
python3 video_logo_remover.py input.mp4 output.mp4 \
  --roi 40,30,180,90 \
  --removal-mode static_rectangle \
  --inpaint-engine opencv \
  --overwrite
```

Prepare VOID chunk packages without running VOID locally:

```bash
PYTHONPATH=src python3 scripts/run_full_void_pipeline.py \
  --input input.mp4 \
  --output-dir outputs/full_void_pipeline \
  --x 120 --y 80 --width 220 --height 300 \
  --removal-mode ai_object \
  --resource-profile l4_pro_balanced
```

Inspect a generated mask cache:

```bash
PYTHONPATH=src python3 scripts/inspect_mask_cache.py \
  --video input.mp4 \
  --cache-dir outputs/full_void_pipeline/mask_cache/<cache_key> \
  --output-dir outputs/mask_cache_inspection
```

Create a before/after comparison video:

```bash
python3 video_edit.py \
  --before input.mp4 \
  --after output.mp4 \
  --out comparison.mp4 \
  --circle 520,340,36 \
  --target-duration 12 \
  --canvas 1080x1350
```

## Colab VOID Pass 1

Use `in_colab.ipynb` to run Netflix VOID Pass 1 in Colab. The notebook expects
a prepared VOID package containing:

```text
input_video.mp4
quadmask_0.mp4
prompt.json
manifest.json
```

The notebook does not include model weights. It downloads public dependencies
from the official upstream sources.

## Environment Variables

Use `.env.example` as a template. Important variables include:

```text
REDIS_URL
SAM2_REPO_DIR
SAM2_CHECKPOINT
SAMURAI_REPO_DIR
DAM4SAM_REPO_DIR
SAM2LONG_REPO_DIR
VOID_REPO_DIR
VENHANCER_COMMAND_TEMPLATE
REALESRGAN_COMMAND_TEMPLATE
OLLAMA_BASE_URL
```

## Publishing Notes

The repository intentionally excludes:

- virtual environments,
- model checkpoints and weights,
- generated outputs,
- exported VOID packages,
- local mask caches,
- private local paths,
- `.env` files,
- notebook execution outputs.

Large sample videos should be shared through the Google Drive folder above or
GitHub Releases, not committed directly to the repository.

## License

This project's original code is released under Apache-2.0. Third-party models,
repositories, and packages have their own licenses and terms. See
`THIRD_PARTY_NOTICES.md` before using the full pipeline commercially.
