# RallyReview

Badminton rally video analysis. Upload a rally video (or paste a YouTube link)
and review it side by side with a 2D top-down court showing players, shuttle
position, trajectory and shot-by-shot animation synchronized to the video.

**Status:** Phases 0–8, running locally. Real computer-vision analysis (court,
players, shuttle, shots) drives the 2D view; see [CV pipeline](docs/cv-pipeline.md).
Videos, results and caches are stored locally in SQLite and files. Hosting
(object storage, a job queue, GPU workers) is the next step; see [Roadmap](#roadmap).

## Repository layout

```
frontend/   React + TypeScript + Vite web app (UI, video playback, 2D court)
backend/    Python FastAPI service (uploads, YouTube downloads, analysis jobs, storage)
            and `rallycv`, the CV pipeline
docs/       Architecture notes and the analysis data contract
storage/    Created on first run, git-ignored: database, videos, caches (see Storage)
```

## Tech stack

| Area | Technology |
| ---- | ---------- |
| Frontend | React 19, TypeScript (strict), Vite; CSS Modules with design tokens; lucide-react icons. No router, state library or CSS framework |
| Backend | Python 3.11+, FastAPI, Uvicorn, Pydantic (camelCase JSON) |
| Computer vision | PyTorch, Hugging Face Transformers, OpenCV, NumPy |
| Storage | SQLite (WAL mode) plus content-addressed video files and detection caches in `storage/` |
| YouTube | yt-dlp, downloading the video stream only (H.264 MP4, up to 1080p), so no ffmpeg is needed |
| Tests and linting | Vitest, Testing Library and oxlint (frontend); pytest and ruff (backend) |

## ML models

Two pretrained neural networks do the detection; everything after them is
geometry and rules (see [CV pipeline](docs/cv-pipeline.md)).

| Model | Detects | How it is used | License |
| ----- | ------- | -------------- | ------- |
| [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r18vd) (`PekingU/rtdetr_v2_r18vd`, ResNet-18 backbone, COCO) | Players (the "person" class) | Run through Hugging Face Transformers 7.5 times a second, on the part of the frame around the court; boxes are interpolated in between | Apache-2.0 |
| [TrackNetV3](https://github.com/qaz812345/TrackNetV3) (authors' pretrained weights) | The shuttle | Sees 8 consecutive frames plus the court background at 512×288 and outputs a heatmap per frame. The network is re-implemented for inference in `rallycv/tracknet.py` | MIT |

Not learned: court detection (white-line detection and a fit of the BWF court
model), the 3D camera, hits, landings, who hit, and shot types (rules). Without
the TrackNet weights, a classical motion-based shuttle detector is used instead.
`make models` downloads both models into `backend/data/models`.

## Requirements

- Node.js 20+ (developed on 22)
- Python 3.11+ (developed on 3.14)
- ~1 GB disk for PyTorch and model weights; CUDA or Apple Silicon (MPS) recommended
- For YouTube links: Node.js (or Deno) on the PATH; yt-dlp is installed by `make setup`

## Getting started

```sh
make setup          # npm install, Python venv (incl. PyTorch) and model weights

make dev-backend    # terminal 1: API on http://127.0.0.1:8000
make dev-frontend   # terminal 2: app on http://localhost:5173 (proxies /api to the backend)
```

Open http://localhost:5173 and upload a rally filmed from a fixed camera behind
the baseline, or paste a YouTube link. Toggle **Overlay** on the video to see what
was detected. If the court isn't found, you'll be asked to click its four corners.

Up to 11 minutes of video is analysed at a time; for a longer YouTube video you
choose the part to analyse. Uploading or linking the same video again returns the
earlier result instantly (see [Storage](#storage)).

For UI work without the models: `make dev-backend-mock` (simulated data).

On Windows the Makefile's `.venv/bin/...` paths don't exist (the venv uses
`.venv\Scripts\`). Run the tools through Python rather than their `.exe`
launchers. Windows Smart App Control blocks those unsigned launchers
("An Application Control policy has blocked this file"):

```powershell
# Each block starts in the project root.

# one-time setup
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,cv,youtube]"
.venv\Scripts\python scripts\download_models.py
cd ..\frontend
npm install

# terminal 1 (backend)
cd backend
.venv\Scripts\python -m uvicorn app.main:app --reload --reload-dir app --reload-dir rallycv --port 8000

# terminal 2 (frontend)
cd frontend
npm run dev

# tests
cd backend; .venv\Scripts\python -m pytest; cd ..
cd frontend; npm test
```

## Using the app

- **Choose a video:** drop a file on the **Upload a file** tab, or paste a link on the
  **YouTube link** tab, press **Check link**, then **Analyse video**. For YouTube
  videos over 11 minutes, pick the part to analyse (e.g. `2:00` to `13:00`).
- **Review:** the court, the current-shot strip and the rally timeline follow the
  video. Click a shot in the list or on the timeline to jump to it.
- **Playback speed:** 0.25x, 0.5x, 0.75x, 1x, 1.5x and 2x. The court view stays in
  sync at every speed.
- **Overlay:** draws the detected court, players and shuttle on the video; the
  quickest way to check an analysis.
- **Mark court corners:** if the court isn't found (or is found wrongly), click its four
  corners and the video is re-analysed with them. Adjusting a court that was found
  reuses the cached detections, so it takes seconds. If the court wasn't found at
  all, the shuttle is tracked while you mark the corners, so only the players are
  left to detect (about 45 s instead of about 150 s for a 21 s clip on CPU).

Keyboard, on the video: `Space`/`K` play or pause, `←`/`→` seek, `<`/`>` speed,
`M` mute, `F` fullscreen. On the timeline: arrows ±1 s (`Shift` ±5 s),
`PageUp`/`PageDown` previous/next shot, `Home`/`End`.

## Configuration

The backend is configured entirely through environment variables:

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `RALLYREVIEW_ANALYZER` | `cv` | `cv` runs the real pipeline; `mock` returns simulated data |
| `RALLYREVIEW_STORAGE_DIR` | `storage/` (project root) | Database, videos and caches (see [Storage](#storage)) |
| `RALLYREVIEW_MODELS_DIR` | `backend/data/models` | TrackNet checkpoint (`ckpts/`) and Hugging Face cache (`hf/`) |
| `RALLYREVIEW_PERSON_MODEL` | `PekingU/rtdetr_v2_r18vd` | RT-DETRv2 model id on Hugging Face, or a local path |
| `RALLYREVIEW_DEVICE` | auto (`cuda`, then `mps`, then `cpu`) | Torch device for both models |
| `RALLYREVIEW_MAX_UPLOAD_BYTES` | 2 GiB | Upload and YouTube download size limit |
| `RALLYREVIEW_MAX_VIDEO_SECONDS` | `660` | Longest stretch of video analysed at once (11 min) |
| `RALLYREVIEW_YOUTUBE` | `on` | `off` disables YouTube links |
| `RALLYREVIEW_YOUTUBE_MAX_SOURCE_SECONDS` | `10800` | Longest YouTube video that may be downloaded (3 h) |
| `RALLYREVIEW_MOCK_STAGE_SECONDS` | `1.2` | Simulated stage duration (mock only) |

The frontend dev server proxies `/api` to `RALLYREVIEW_BACKEND_URL`
(default `http://127.0.0.1:8000`). `make models` (or
`python backend/scripts/download_models.py`) honours `RALLYREVIEW_MODELS_DIR`
and `RALLYREVIEW_PERSON_MODEL`, so weights can be fetched at build time and the
server needs no network access at startup.

## Storage

Everything the app stores lives in `storage/` at the project root (git-ignored):

```
storage/
  rallyreview.sqlite3     jobs, videos, YouTube downloads, cached results
  videos/<sha256>.mp4     each distinct video once, named by its content
  cache/<sha256>/         per-video detections, so re-runs skip the neural networks
  tmp/                    uploads and downloads in progress (cleared on start)
```

- **Same video, same settings:** answered from the results table instantly, whatever
  the file is called. A YouTube link already downloaded isn't downloaded again.
- **Same video, new settings** (e.g. manually marked court corners, or a newer
  pipeline version): the cached detections are reused, so only the cheap steps rerun.
- Jobs survive a backend restart; ones that were running are marked as interrupted.
- Deleting `storage/` resets everything; deleting `storage/cache/` only costs time.

### YouTube links

The backend downloads the video stream only (H.264, up to 1080p, no audio), so no
ffmpeg is needed. YouTube requires a JavaScript runtime for yt-dlp: Node.js or Deno
must be on the PATH. Private, members-only, age-restricted and live videos are
refused with a clear message. When hosted, YouTube may block downloads from
datacenter IP addresses; users can always upload the file instead. Downloading
from YouTube is subject to YouTube's terms and the footage's copyright.

## Checks

```sh
make test     # vitest + pytest (fast)
make test-cv  # slow: full pipeline on real footage (needs backend/data/samples)
make lint     # oxlint + ruff
make build    # type-check and production build of the frontend
make check    # all of the above
```

## Docs

- [Architecture](docs/architecture.md): components, boundaries and data flow
- [Analysis contract](docs/analysis-contract.md): the trajectory data format
  the frontend consumes
- [CV pipeline](docs/cv-pipeline.md): how analysis works, accuracy, limitations
- [Third-party notices](THIRD_PARTY_NOTICES.md): TrackNetV3 (MIT), RT-DETRv2 (Apache-2.0), yt-dlp (Unlicense)

## Roadmap

| Phase | Scope |
| ----- | ----- |
| 0 | Project setup |
| 1 | UI shell: sidebar, header, video and trajectory panels, empty states |
| 2 | Video upload and preview |
| 3 | Mock processing flow (Upload → Processing → Ready) with mock analysis data |
| 4 | 2D court: lines, net, players, shuttle, trajectory, shot indicators |
| 5 | Court animation synchronized with video playback |
| 6 | Computer vision: court detection, player tracking, shuttle tracking, shot detection, video overlay, manual court marking |
| 7 | Accuracy and speed: hitter assignment across rallies, serve detection, cuts in edited footage, shuttle into the net, ~2.4x faster on CPU; playback speeds 0.25x to 2x |
| 8 | Local storage and YouTube: SQLite + content-addressed files in `storage/`, result and detection caches, YouTube links, 11-minute limit, jobs survive restarts |
| next | Hosting: Postgres, object storage (S3/R2), a job queue with GPU workers, sign-in; a library of past analyses. Accuracy: 3D trajectory fitting, learned shot types |
