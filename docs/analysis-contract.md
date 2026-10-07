# Analysis contract (schema v1, draft)

The data the frontend needs to render one analysed rally. Source of truth for
the frontend: [`frontend/src/types/analysis.ts`](../frontend/src/types/analysis.ts).
Backend mirror: [`backend/app/analysis/contract.py`](../backend/app/analysis/contract.py)
(Pydantic). Keep the two in sync; the backend tests check the camelCase wire format.

This is a draft. It will be refined in Phases 3–5 as the mock and the
visualization exercise it.

## Coordinate system

All positions are **normalized court coordinates** on the full doubles court
(6.10 m × 13.40 m), independent of video resolution or screen layout.

```
            x = 0                 x = 1
   y = 1  ┌───────────────────────┐   Player B baseline
          │                       │
          │       Player B        │
          │                       │
   y = 0.5├───────── net ─────────┤
          │                       │
          │       Player A        │
          │                       │
   y = 0  └───────────────────────┘   Player A baseline
```

- `x`: across the width; 0 = left sideline, 1 = right sideline, viewed from
  Player A's baseline.
- `y`: along the length; 0 = Player A's baseline, 1 = Player B's baseline.
  The net is at `y = 0.5`.
- Values may fall slightly outside `[0, 1]` (shuttle or player out of bounds).
- Whether the court is drawn portrait or landscape is purely a rendering choice.
- Positions are ground-plane projections. Shuttle height is out of scope for 2D.

## Shape

```ts
RallyAnalysis {
  schemaVersion: 1
  source?: 'mock' | 'cv'
  calibration?: { courtToImage: number[9], confidence, manual } // court -> image homography
  durationSec: number
  players: PlayerTrack[]       // exactly two for singles: A and B
  shuttle: ShuttleSample[]     // time-ordered
  shots: Shot[]                // time-ordered, non-overlapping
}

PlayerTrack   { id: 'A' | 'B', label: string, samples: { t, position }[] }
ShuttleSample { t, position: { x, y } | null,   // floor position; null = unknown
                image?: { x, y } | null,       // where it was detected in the video frame
                height?: number | null }       // metres, when reconstructed
Shot          { index, hitter: 'A' | 'B', type, startTime, endTime }
ShotType      'serve' | 'clear' | 'drop' | 'smash' | 'drive' | 'net' | 'lift' | 'unknown'
```

All times are seconds from the start of the video, so the UI can look up state
directly from `video.currentTime`.

## Design notes

- `position: null` on shuttle samples is deliberate. Real trackers lose the
  shuttle; the UI must handle gaps instead of the producer inventing positions.
- Samples need not be at a fixed rate. The UI interpolates between samples.
- Doubles (four players) would extend `PlayerId`; not in scope for v1.
- A clip may contain several rallies: each starts with a `serve` shot, and
  there are no shots during dead time between them.
- For CV output, floor positions mid-flight are estimates (a single camera
  can't measure height); `image` is what was actually detected.
