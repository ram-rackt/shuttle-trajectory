/**
 * Time-based lookups into analysis data. Pure functions, so the court can be
 * rendered for any instant (Phase 5 drives this from video.currentTime).
 */

import type {
  CourtPoint,
  ImagePoint,
  PlayerSample,
  RallyAnalysis,
  Shot,
  ShuttleSample,
} from '../../types/analysis'

/** Index of the last item whose key <= time, or -1 if time is before the first item. */
function indexAtOrBefore<T>(
  items: readonly T[],
  time: number,
  key: (item: T) => number,
): number {
  let lo = 0
  let hi = items.length - 1
  let found = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (key(items[mid]) <= time) {
      found = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return found
}

function lerp(a: CourtPoint, b: CourtPoint, u: number): CourtPoint {
  return { x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u }
}

/** Player position at `time`, linearly interpolated and clamped to the track's range. */
export function playerPositionAt(samples: readonly PlayerSample[], time: number): CourtPoint | null {
  if (samples.length === 0) return null
  const i = indexAtOrBefore(samples, time, (s) => s.t)
  if (i === -1) return samples[0].position
  if (i === samples.length - 1) return samples[i].position
  const a = samples[i]
  const b = samples[i + 1]
  const span = b.t - a.t
  return span > 0 ? lerp(a.position, b.position, (time - a.t) / span) : a.position
}

/**
 * Shuttle position at `time`. Null outside the tracked range or inside a
 * detection gap: we never invent positions the tracker didn't report.
 */
export function shuttlePositionAt(samples: readonly ShuttleSample[], time: number): CourtPoint | null {
  const i = indexAtOrBefore(samples, time, (s) => s.t)
  if (i === -1) return null
  const a = samples[i]
  if (i === samples.length - 1 || a.t === time) return a.position
  const b = samples[i + 1]
  if (!a.position || !b.position) return null
  return lerp(a.position, b.position, (time - a.t) / (b.t - a.t))
}

/**
 * Index of the shot in play at `time`: the last shot whose startTime <= time.
 * Null before the first shot.
 */
export function shotIndexAt(shots: readonly Shot[], time: number): number | null {
  const i = indexAtOrBefore(shots, time, (s) => s.startTime)
  return i === -1 ? null : i
}

export interface ShotPath {
  shotIndex: number
  /** Contiguous runs of tracked positions; split wherever the shuttle was lost. */
  segments: CourtPoint[][]
  /** Where the shuttle was struck (first tracked point of the shot, if any). */
  hitPoint: CourtPoint | null
  /** Where the shuttle ended up at the end of the shot. */
  endPoint: CourtPoint | null
}

/** Splits the shuttle track into per-shot paths for drawing. */
export function buildShotPaths(analysis: RallyAnalysis): ShotPath[] {
  return analysis.shots.map((shot, shotIndex) => {
    const segments: CourtPoint[][] = []
    let run: CourtPoint[] = []
    for (const sample of analysis.shuttle) {
      if (sample.t < shot.startTime || sample.t > shot.endTime) continue
      if (sample.position) {
        run.push(sample.position)
      } else if (run.length) {
        segments.push(run)
        run = []
      }
    }
    if (run.length) segments.push(run)

    const hitter = analysis.players.find((p) => p.id === shot.hitter)
    const hitPoint =
      shuttlePositionAt(analysis.shuttle, shot.startTime) ??
      (hitter ? playerPositionAt(hitter.samples, shot.startTime) : null)
    const endPoint = shuttlePositionAt(analysis.shuttle, shot.endTime) ?? segments.at(-1)?.at(-1) ?? null
    return { shotIndex, segments, hitPoint, endPoint }
  })
}

/**
 * Tracked shuttle positions over the last `windowSec` seconds up to `time`,
 * ending at the current position. Split into runs at detection gaps.
 */
export function shuttleTrail(
  samples: readonly ShuttleSample[],
  time: number,
  windowSec: number,
): CourtPoint[][] {
  const end = indexAtOrBefore(samples, time, (s) => s.t)
  if (end === -1) return []
  const start = Math.max(indexAtOrBefore(samples, time - windowSec, (s) => s.t), 0)

  const segments: CourtPoint[][] = []
  let run: CourtPoint[] = []
  for (let i = start; i <= end; i++) {
    const p = samples[i].position
    if (p) {
      run.push(p)
    } else if (run.length) {
      segments.push(run)
      run = []
    }
  }
  const current = shuttlePositionAt(samples, time)
  if (current && run.length) run.push(current)
  if (run.length > 1) segments.push(run)
  return segments.filter((s) => s.length > 1)
}

/** Seconds of the current shot after which "previous" restarts it rather than going back. */
const RESTART_THRESHOLD_SEC = 0.5
/** Small nudge so a seek lands inside the target shot despite float rounding. */
const SHOT_SEEK_EPSILON = 0.001

/** Start time to seek to for the next shot, or null if there is none. */
export function nextShotTime(shots: readonly Shot[], time: number): number | null {
  const next = shots.find((s) => s.startTime > time + SHOT_SEEK_EPSILON)
  return next ? next.startTime + SHOT_SEEK_EPSILON : null
}

/**
 * Start time to seek to for "previous shot": restarts the current shot if
 * we're well into it (media-player convention), otherwise the one before.
 */
export function previousShotTime(shots: readonly Shot[], time: number): number | null {
  const i = shotIndexAt(shots, time)
  if (i === null) return null
  const target =
    time - shots[i].startTime > RESTART_THRESHOLD_SEC || i === 0 ? shots[i] : shots[i - 1]
  return target.startTime + SHOT_SEEK_EPSILON
}

/** Seek target for a specific shot. */
export function shotSeekTime(shot: Shot): number {
  return shot.startTime + SHOT_SEEK_EPSILON
}

/** Where the shuttle was detected in the video frame at `time` (normalized), if seen. */
export function shuttleImageAt(samples: readonly ShuttleSample[], time: number): ImagePoint | null {
  const i = indexAtOrBefore(samples, time, (s) => s.t)
  if (i === -1) return null
  const a = samples[i]
  if (i === samples.length - 1 || a.t === time) return a.image ?? null
  const b = samples[i + 1]
  if (!a.image || !b.image) return null
  const u = (time - a.t) / (b.t - a.t)
  return { x: a.image.x + (b.image.x - a.image.x) * u, y: a.image.y + (b.image.y - a.image.y) * u }
}

/** Detected image positions over the last `windowSec` up to `time`, split at gaps. */
export function shuttleImageTrail(
  samples: readonly ShuttleSample[],
  time: number,
  windowSec: number,
): ImagePoint[][] {
  const end = indexAtOrBefore(samples, time, (s) => s.t)
  if (end === -1) return []
  const start = Math.max(indexAtOrBefore(samples, time - windowSec, (s) => s.t), 0)
  const runs: ImagePoint[][] = []
  let run: ImagePoint[] = []
  for (let i = start; i <= end; i++) {
    const p = samples[i].image
    if (p) run.push(p)
    else if (run.length) {
      runs.push(run)
      run = []
    }
  }
  const current = shuttleImageAt(samples, time)
  if (current && run.length) run.push(current)
  if (run.length) runs.push(run)
  return runs.filter((r) => r.length > 1)
}
