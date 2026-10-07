/**
 * Badminton court geometry in meters (BWF dimensions), and the mapping from
 * normalized analysis coordinates to the SVG drawing.
 *
 * The court is drawn landscape: Player A's baseline on the left, Player B's on
 * the right. Normalized y (along the length) maps to SVG x; normalized x
 * (across the width, 0 = Player A's left sideline) maps to SVG y, so Player A's
 * left is the top edge of the drawing.
 */

import type { CourtPoint } from '../../types/analysis'

export const COURT = {
  length: 13.4,
  width: 6.1,
  singlesInset: 0.46,
  shortServiceFromNet: 1.98,
  doublesLongServiceFromBaseline: 0.76,
  lineWidth: 0.04,
} as const

export const NET_X = COURT.length / 2

export interface SvgPoint {
  x: number
  y: number
}

/** Normalized court point → SVG coordinates in meters. */
export function toSvg(p: CourtPoint): SvgPoint {
  return { x: p.y * COURT.length, y: p.x * COURT.width }
}

/** SVG `points` attribute for a polyline through normalized points. */
export function toPolylinePoints(points: readonly CourtPoint[]): string {
  return points
    .map((p) => {
      const s = toSvg(p)
      return `${s.x.toFixed(3)},${s.y.toFixed(3)}`
    })
    .join(' ')
}

/** All court lines as [x1, y1, x2, y2] segments, in meters. */
export function courtLines(): [number, number, number, number][] {
  const { length: L, width: W, singlesInset: s } = COURT
  const shortA = NET_X - COURT.shortServiceFromNet
  const shortB = NET_X + COURT.shortServiceFromNet
  const longA = COURT.doublesLongServiceFromBaseline
  const longB = L - COURT.doublesLongServiceFromBaseline
  const mid = W / 2
  return [
    // Outer boundary (doubles sidelines and baselines)
    [0, 0, L, 0],
    [0, W, L, W],
    [0, 0, 0, W],
    [L, 0, L, W],
    // Singles sidelines
    [0, s, L, s],
    [0, W - s, L, W - s],
    // Short service lines
    [shortA, 0, shortA, W],
    [shortB, 0, shortB, W],
    // Doubles long service lines
    [longA, 0, longA, W],
    [longB, 0, longB, W],
    // Centre lines (baseline to short service line, each half)
    [0, mid, shortA, mid],
    [shortB, mid, L, mid],
  ]
}
