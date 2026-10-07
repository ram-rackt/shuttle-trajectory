# Architecture

## Goal

Keep the UI completely independent of how analysis is produced. The frontend
consumes a single data contract ([analysis-contract.md](analysis-contract.md));
the backend produces it, first from mock data and later from the CV pipeline.

## Components

```
┌────────────── frontend (React/TS) ──────────────┐      ┌──────────── backend (FastAPI) ────────────┐
│                                                 │      │                                            │
│  UI shell ─ Video panel ─ 2D court panel        │      │  API layer (app/)                          │
│        │                     ▲                  │ HTTP │    uploads, jobs, status, results          │
│        └── playback time ────┘                  │◄────►│            │                               │
│                                                 │ /api │            ▼                               │
│  api/client.ts   types/analysis.ts (contract)   │      │  Analysis provider                         │
│                                                 │      │    Phase 3: mock generator                 │
└─────────────────────────────────────────────────┘      │    Phase 6+: CV pipeline (separate pkg)    │
                                                         └────────────────────────────────────────────┘
```

### Frontend (`frontend/`)

- React 19 + TypeScript (strict) + Vite. Tests with Vitest + Testing Library.
- `src/types/analysis.ts`: TypeScript form of the analysis contract. UI code
  depends on this, never on CV internals.
- `src/api/client.ts`: the only place that talks HTTP. Components call
  functions here, which makes the transport easy to mock.
- In dev, Vite proxies `/api/*` to the backend, so no backend URL is hard-coded.

- `src/features/analysis/useAnalysis.ts`: the Upload → Processing → Ready
  state machine. Starts automatically when a video is loaded, polls job status,
  supports cancel/retry, and discards results that belong to a previous video.
- `src/features/court/`: the 2D court. `CourtView` is a pure function of
  `(analysis, time)` drawn as SVG in meters (BWF dimensions; `geometry.ts` maps
  normalized coordinates to the drawing). `sampling.ts` holds the time lookups
  (interpolated player/shuttle positions, current shot, per-shot paths split at
  tracking gaps). Static layers are memoized so only players and the shuttle
  re-render as time changes.
- Playback sync: `App` owns the `<video>` ref and `useVideoPlayer`, which samples
  `currentTime` every animation frame while playing. That single clock drives
  the court, the current-shot strip and the timeline; all seeking (scrubber,
  timeline drag, shot list, prev/next, keyboard) goes through `player.seek`, so
  the video stays the source of truth.
- `src/features/review/RallyTimeline.tsx`: shot blocks, playhead, drag-to-scrub,
  keyboard slider (arrows ±1 s, Shift ±5 s, PageUp/PageDown prev/next shot,
  Home/End) and a clickable shot list.

### Backend (`backend/`)

- FastAPI, run with uvicorn. Tests with pytest; lint/format with ruff.
- `app/api/analyses.py`: HTTP endpoints for analysis jobs (below).
- `app/jobs.py`: in-memory job store and the job runner (runs in a worker thread).
- `app/analysis/contract.py`: Pydantic form of the analysis contract
  (snake_case in Python, camelCase on the wire).
- `app/analysis/provider.py`: the `AnalysisProvider` protocol. Anything that
  implements `analyze(video_path, duration_sec, on_progress) -> RallyAnalysis`
  can be plugged in via `create_app(provider=...)`.
- `app/analysis/mock.py`: `MockAnalysisProvider`, which simulates the pipeline
  stages and generates a deterministic, plausible rally.
- `app/analysis/cv_provider.py`: `CVAnalysisProvider`, the adapter that runs
  the `rallycv` pipeline and converts its output to the contract. Selected by
  default; `RALLYREVIEW_ANALYZER=mock` selects the mock.
- `rallycv/`: the computer-vision pipeline, independent of the web app (no
  imports from `app`). See [cv-pipeline.md](cv-pipeline.md).

### Analysis API

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `POST` | `/api/analyses` | Multipart upload (`video`, optional `duration_sec`). Returns `202` with job status. |
| `GET` | `/api/analyses/{id}` | Job status: `queued` / `processing` / `ready` / `failed`, `progress` (0–1), `stages`, `currentStage`, `error`. |
| `GET` | `/api/analyses/{id}/result` | The `RallyAnalysis` once ready (`409` before that). |
| `POST` | `/api/analyses/{id}/calibration` | Re-run on the same video with 4 manually marked court corners (`{corners: [{x, y}×4]}`, normalized, near-left → near-right → far-right → far-left). Returns a new job. |

Failed jobs carry `errorCode` (e.g. `court_not_found`, `shuttle_not_found`,
`court_not_visible`, `video_too_long`) so the UI can offer the right fix.

Uploads are stored under `backend/data/uploads/` (git-ignored). Config via env:
`RALLYREVIEW_DATA_DIR`, `RALLYREVIEW_MAX_UPLOAD_BYTES` (default 2 GiB),
`RALLYREVIEW_MOCK_STAGE_SECONDS` (default 1.2).

The frontend sends `duration_sec` (read from the video metadata in the browser)
so the mock can fit the rally to the clip. The real pipeline will read the
duration from the file itself and can ignore it.

## Planned CV pipeline (Phase 6+)

```
Video → court detection → player tracking → shuttle tracking → shot detection
      → court-coordinate transform (homography) → RallyAnalysis (normalized coords)
```

Each stage should be independently testable on recorded fixtures. Shuttle
tracking is expected to be the hardest stage and is deliberately left until the
prototype works end-to-end on mock data.

## Key decisions

| Decision | Reason |
| -------- | ------ |
| Separate frontend and Python backend | CV ecosystem (OpenCV, PyTorch) is Python. The frontend stays a plain web app. |
| Normalized court coordinates (0–1) | Rendering is decoupled from video resolution, camera angle and screen size. |
| Contract defined up front, versioned | Mock and real CV must produce identical shapes; `schemaVersion` allows evolution. |
| Vite dev proxy for `/api` | Same-origin requests in dev and prod; no CORS config yet. |
| No state library, router or CSS framework yet | Not needed for the prototype; add when a phase demands it. |
| Async jobs with polling | CV will take seconds to minutes per rally; the request must not block. Polling is simpler than WebSockets/SSE at this stage. |
| In-memory job store, background threads | Fine for a single-process prototype. Jobs are lost on restart; production needs a persistent queue and separate workers. |
