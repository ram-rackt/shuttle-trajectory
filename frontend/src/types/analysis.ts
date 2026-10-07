/**
 * RallyReview analysis contract (schema v1, DRAFT).
 *
 * This is the only shape the frontend knows about. Whatever produces it —
 * the Phase 3 mock or the real CV pipeline — must emit this structure.
 * See docs/analysis-contract.md for the full description.
 *
 * Coordinate system (normalized, 0–1, full doubles court):
 *   x: across the court width.  0 = left sideline, 1 = right sideline
 *      (as seen from Player A's baseline looking toward the net).
 *   y: along the court length. 0 = Player A's baseline, 1 = Player B's baseline.
 *      The net is at y = 0.5.
 * How the court is drawn on screen (portrait/landscape) is a rendering concern.
 */

export const ANALYSIS_SCHEMA_VERSION = 1

/** Normalized court position. Values may slightly exceed [0, 1] when out of bounds. */
export interface CourtPoint {
  x: number
  y: number
}

export type PlayerId = 'A' | 'B'

export interface PlayerSample {
  /** Seconds from the start of the video. */
  t: number
  position: CourtPoint
}

export interface PlayerTrack {
  id: PlayerId
  label: string
  samples: PlayerSample[]
}

/** Normalized video-frame position: x, y in [0, 1] from the top-left corner. */
export interface ImagePoint {
  x: number
  y: number
}

export interface ShuttleSample {
  /** Seconds from the start of the video. */
  t: number
  /** Floor (top-down) position; null when unknown at this time. */
  position: CourtPoint | null
  /** Where the shuttle was detected in the video frame; null/absent when not seen. */
  image?: ImagePoint | null
  /** Estimated height above the floor in metres, when reconstructed. */
  height?: number | null
}

export type ShotType =
  | 'serve'
  | 'clear'
  | 'drop'
  | 'smash'
  | 'drive'
  | 'net'
  | 'lift'
  | 'unknown'

export interface Shot {
  /** 0-based order within the rally. */
  index: number
  hitter: PlayerId
  type: ShotType
  /** Time the shuttle is struck (seconds). */
  startTime: number
  /** Time of the next strike, or rally end (seconds). */
  endTime: number
}

export interface CameraCalibration {
  /**
   * Row-major 3x3 homography from normalized court coordinates (x, y in 0-1)
   * to normalized image coordinates (0-1). Used to draw the court on the video.
   */
  courtToImage: number[]
  /** Fraction of the court's painted lines confirmed in the image (0-1). */
  confidence: number
  /** True if the court was marked by the user rather than detected. */
  manual: boolean
}

export interface RallyAnalysis {
  schemaVersion: typeof ANALYSIS_SCHEMA_VERSION
  /** 'cv' for real computer-vision output, 'mock' for simulated data. */
  source?: 'mock' | 'cv'
  /** Duration of the analysed video in seconds. */
  durationSec: number
  players: PlayerTrack[]
  /** Time-ordered shuttle samples. */
  shuttle: ShuttleSample[]
  /** Time-ordered, non-overlapping shots. A rally starts with a serve. */
  shots: Shot[]
  /** Present for CV output: lets the UI overlay the detected court on the video. */
  calibration?: CameraCalibration | null
}
