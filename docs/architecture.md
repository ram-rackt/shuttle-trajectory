# Architecture

## Goal

Keep the UI completely independent of how analysis is produced. The frontend
consumes a single data contract ([analysis-contract.md](analysis-contract.md));
the backend produces it, from the CV pipeline or from a mock generator.

## Components

```
┌────────────── frontend (React/TS) ──────────────┐      ┌──────────── backend (FastAPI) ────────────┐
│                                                 │      │                                            │
│  Video source: file upload or YouTube link      │      │  API layer (app/api/)                      │
│  UI shell ─ Video panel ─ 2D court panel        │ HTTP │    uploads, YouTube, jobs, results, media  │
│        │                     ▲                  │◄────►│            │                               │
│        └── playback time ────┘                  │ /api │            ▼                               │
│                                                 │      │  Job runner ── YouTube download (yt-dlp)   │
│  api/client.ts   types/analysis.ts (contract)   │      │       │                                    │
│                                                 │      │       ▼                                    │
└─────────────────────────────────────────────────┘      │  Analysis provider: CV pipeline | mock     │
                                                         │       │                                    │
                                                         │       ▼                                    │
                                                         │  Storage: SQLite + files in storage/       │
                                                         └────────────────────────────────────────────┘
```

### Frontend (`frontend/`)

- React 19 + TypeScript (strict) + Vite. Tests with Vitest + Testing Library.
- `src/types/analysis.ts`: TypeScript form of the analysis contract. UI code
  depends on this, never on CV internals.
- `src/api/client.ts`: the only place that talks HTTP. Components call
  functions here, which makes the transport easy to mock. Errors carry the
  server's `code` (`ApiError.code`).
- In dev, Vite proxies `/api/*` to the backend, so no backend URL is hard-coded.
- `src/hooks/useServerConfig.ts`: reads `/api/config` once (upload size limit,
  longest video, whether YouTube links are enabled). Missing or malformed fields
  keep safe defaults, so an older backend can't break the UI.
- `src/features/video/useVideoFile.ts`: `useVideoSource`, the current video
  source: a local file (played from a blob URL) or a YouTube video (played from
  the server's copy once downloaded). `YouTubeForm.tsx` checks a link, shows its
  title, length and thumbnail, and asks for a part (`youtubeRange.ts`) when the
  video is longer than the analysis limit.
- `src/features/analysis/useAnalysis.ts`: the Upload → Processing → Ready
  state machine. Starts automatically when a video is chosen, polls job status,
  supports cancel (also stopping the job on the server) and retry, and discards
  results that belong to a previous video. It remembers the server's copy of
  the video once a job reports it, so a YouTube video stays playable (and its
  court can be marked) after a failure, a cancel or a re-run. `playbackUrlFor`
  and `downloadProgressFor` derive what to play and how far a YouTube download is.
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
  the video stays the source of truth. Playback speed (0.25x to 2x, `<` / `>`)
  is set on the video element, so the court and timeline follow it with no
  extra wiring; the chosen speed is re-applied when a new video loads.
- `src/features/review/RallyTimeline.tsx`: shot blocks, playhead, drag-to-scrub,
  keyboard slider (arrows ±1 s, Shift ±5 s, PageUp/PageDown prev/next shot,
  Home/End) and a clickable shot list.

### Backend (`backend/`)

- FastAPI, run with uvicorn. Tests with pytest; lint/format with ruff.
- `app/config.py`: settings from environment variables (full list in the README).
- `app/main.py`: app factory; on startup creates the storage, the job store and
  the job runner, and serves `/api/health` and `/api/config`.
- `app/api/analyses.py`: HTTP endpoints for analysis jobs (below).
- `app/api/uploads.py`: reads an upload's multipart body as it arrives, straight
  into `storage/tmp/`, hashing it on the way: never held in memory, written once,
  and stopped as soon as it passes the size limit (a declared size over the limit
  is refused before anything is read). Fields sent before the video (the web app
  sends `job_id` and `duration_sec` first) are checked before any of it is stored.
- `app/api/media.py`: serves stored videos (with range requests, for seeking)
  and checks YouTube links.
- `app/storage.py`: `Storage`, the only module that touches `storage/`: the
  SQLite database and the content-addressed video files.
- `app/youtube.py`: YouTube links via yt-dlp: link parsing, metadata checks
  (guardrails), a cancellable download, and yt-dlp errors translated into
  stable error codes with user-facing messages.
- `app/jobs.py`: `JobStore` (jobs persisted in SQLite) and `JobRunner`, which
  runs jobs in worker threads: download (YouTube only, one at a time), result
  cache check, analysis, and saving the result.
- `app/analysis/contract.py`: Pydantic form of the analysis contract
  (snake_case in Python, camelCase on the wire).
- `app/analysis/provider.py`: the `AnalysisProvider` protocol. Anything that
  implements `analyze(request: AnalysisRequest, on_progress) -> RallyAnalysis`
  can be plugged in via `create_app(provider=...)`. The request carries the
  video path, optional manual court corners, an optional time window and a
  cache directory. A provider's `version` is part of the result cache key.
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
| `POST` | `/api/analyses` | Multipart upload (`video`, optional `duration_sec`, `court_corners`, `job_id`). Returns `202` with job status (already `ready`, `cached: true`, if this video was analysed before). |
| `POST` | `/api/analyses/youtube` | `{url, startSec?, endSec?, jobId?}`: download a YouTube video, then analyse it (or the chosen part). Returns `202` with job status; problems with the link are `4xx` with a `code`. |
| `GET` | `/api/analyses/{id}` | Job status: `queued` / `processing` / `ready` / `failed`, `progress` (0–1), `stages`, `currentStage`, `error`, `errorCode`, plus `video` (the stored video, once available), `window` (the analysed part) and `cached`. |
| `GET` | `/api/analyses/{id}/result` | The `RallyAnalysis` once ready (`409` before that). |
| `POST` | `/api/analyses/{id}/cancel` | Stops a queued or running job (download or analysis). `404` if there is no such job yet; see *Client-chosen job ids* below. |
| `POST` | `/api/analyses/{id}/calibration` | Re-run on the same video with 4 manually marked court corners (`{corners: [{x, y}×4], jobId?}`, normalized, near-left → near-right → far-right → far-left). Returns a new job. |
| `GET` | `/api/youtube/info?url=` | Checks a YouTube link: title, channel, length, resolution, thumbnail, and whether a part must be chosen. |
| `GET` | `/api/videos/{sha256}/file` | The stored video, with HTTP range support (seeking). |
| `GET` | `/api/config` | Limits the UI enforces up front (upload size, video length, YouTube on/off). |
| `GET` | `/api/health` | Liveness and version. |

Errors are JSON `{detail, code}`; failed jobs carry `errorCode`, so the UI can
offer the right fix:

- Analysis: `court_not_found`, `court_not_visible`, `shuttle_not_found`,
  `players_not_found`, `invalid_court_corners`, `video_unreadable`,
  `video_too_long`, `video_too_short`, `video_missing`.
- YouTube: `invalid_url`, `unsupported_site`, `video_private`,
  `video_unavailable`, `video_restricted`, `age_restricted`, `geo_blocked`,
  `live_video`, `unknown_duration`, `source_too_long`, `low_resolution`,
  `too_large`, `no_suitable_format`, `youtube_blocked`, `rate_limited`,
  `network_error`, `download_failed`, `youtube_unavailable` (YouTube support off
  or yt-dlp not installed).
- Time window: `range_required`, `invalid_range`, `range_too_long`.
- Jobs: `cancelled`, `interrupted` (the backend restarted while it ran).

The frontend sends `duration_sec` (read from the video metadata in the browser)
so a video over the limit is refused before it is uploaded, and so the mock can
fit its rally to the clip. The CV pipeline reads the duration from the file
itself and enforces the limit again.

## Job lifecycle

```
upload ──► stream to tmp/, hash ──► videos/<sha256>.mp4 ─┐
                                                         ├─► result cached? ── yes ──► ready (cached)
YouTube link ──► checks ──► download (job stage 0) ──────┘          │
                                                                    no
                                                                    ▼
                                         analysis (rallycv, detections cached per video)
                                                                    ▼
                                                    result saved ──► ready
```

- Videos are identified by the SHA-256 of their content, so the same file under
  any name, or the same YouTube video linked twice, is stored once.
- A result is cached under (video, provider version, settings: manual corners
  and time window). Asking again with the same key returns it immediately.
- With a new key (new corners, a new pipeline version), the expensive
  detections are still reused from `storage/cache/<sha256>/`; see
  [cv-pipeline.md](cv-pipeline.md#analysis-window-and-caching).
- Jobs are rows in SQLite, so status and results survive a restart. Jobs that
  were running when the backend stopped are marked `interrupted`; `tmp/` is
  cleared on startup.
- YouTube downloads run one at a time; analyses run in worker threads.
- When a job fails with `court_not_found`, the runner goes on to precompute the
  shuttle detections (they don't need the court) while the user marks the corners,
  so the re-run is quicker. It stops as soon as a job for another video is waiting.

**Client-chosen job ids.** The create endpoints accept an optional job id (32
lowercase hex digits; the web app always sends one), so the client knows the id
before the request returns. If the user gives up while the request is still being
handled (a YouTube link takes seconds to check), the client cancels that id
anyway: a cancel for a job that doesn't exist yet is remembered for 10 minutes,
and the job is then created already cancelled instead of running for nobody.
Reusing an id is `409`.

## Key decisions

| Decision | Reason |
| -------- | ------ |
| Separate frontend and Python backend | CV ecosystem (OpenCV, PyTorch) is Python. The frontend stays a plain web app. |
| Normalized court coordinates (0–1) | Rendering is decoupled from video resolution, camera angle and screen size. |
| Contract defined up front, versioned | Mock and real CV must produce identical shapes; `schemaVersion` allows evolution. |
| Vite dev proxy for `/api` | Same-origin requests in dev and prod; no CORS config yet. |
| No state library, router or CSS framework yet | Not needed so far; add when a feature demands it. |
| Async jobs with polling | CV takes seconds to minutes per rally; the request must not block. Polling is simpler than WebSockets/SSE at this stage. |
| SQLite + local files behind one storage module | Simple for one machine; the same interface can be backed by Postgres and object storage (S3/R2) when hosted. |
| Videos identified by content hash | The same file under any name is stored and analysed once. |
| YouTube videos downloaded, not streamed | The CV pipeline needs random access to frames and the player needs a stable URL; a stored copy gives both, and caching by content works the same as for uploads. |
| Long videos analysed in a window, not trimmed | No ffmpeg dependency and no re-encoding; the player keeps the original timeline, so result times match the video. |
| Jobs run in background threads | Fine for one machine. Hosted, a queue (e.g. Redis) and separate GPU workers replace them. |
