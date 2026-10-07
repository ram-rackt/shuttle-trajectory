# RallyReview

Badminton rally video analysis. Upload a rally video and review it side by side
with a 2D top-down court showing players, shuttle position, trajectory and
shot-by-shot animation synchronized to the video.

**Status:** Phases 0–6. Real computer-vision analysis (court, players, shuttle, shots) drives the 2D view; see [CV pipeline](docs/cv-pipeline.md). See [Roadmap](#roadmap).

## Repository layout

```
frontend/   React + TypeScript + Vite web app (UI, video playback, 2D court)
backend/    Python FastAPI service (uploads, analysis jobs) and `rallycv`, the CV pipeline
docs/       Architecture notes and the analysis data contract
```

## Requirements

- Node.js 20+ (developed on 22)
- Python 3.11+ (developed on 3.14)
- ~1 GB disk for PyTorch and model weights; Apple Silicon (MPS) or CUDA recommended

## Getting started

```sh
make setup          # npm install, Python venv (incl. PyTorch) and model weights

make dev-backend    # terminal 1: API on http://127.0.0.1:8000
make dev-frontend   # terminal 2: app on http://localhost:5173 (proxies /api to the backend)
```

Open http://localhost:5173 and upload a rally filmed from a fixed camera behind
the baseline. Toggle **Overlay** on the video to see what was detected. If the
court isn't found, you'll be asked to click its four corners.

For UI work without the models: `make dev-backend-mock` (simulated data).

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
- [Third-party notices](THIRD_PARTY_NOTICES.md): TrackNetV3 (MIT), YOLO (AGPL-3.0)

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
| next | Accuracy: 3D trajectory fitting, learned shot types; production infrastructure |
