import { memo } from 'react'
import { applyHomography } from '../../lib/homography'
import type { CameraCalibration, PlayerId, RallyAnalysis } from '../../types/analysis'
import { COURT, courtLines } from '../court/geometry'
import { playerPositionAt, shuttleImageAt, shuttleImageTrail } from '../court/sampling'
import styles from './VideoOverlay.module.css'

const TRAIL_SECONDS = 0.4

interface VideoOverlayProps {
  analysis: RallyAnalysis
  calibration: CameraCalibration
  time: number
  videoWidth: number
  videoHeight: number
}

/**
 * Draws what the analysis "sees" on top of the video: the detected court
 * lines, where each player's feet were placed on the floor, and the shuttle
 * where it was detected. If these line up with the picture, the 2D court view
 * is in sync with the video.
 *
 * The SVG uses the video's native pixel size as its viewBox with
 * xMidYMid meet, which mirrors the video's object-fit: contain exactly.
 */
export function VideoOverlay({
  analysis,
  calibration,
  time,
  videoWidth,
  videoHeight,
}: VideoOverlayProps) {
  if (videoWidth <= 0 || videoHeight <= 0) return null
  const H = calibration.courtToImage
  const toPx = (x: number, y: number) => {
    const p = applyHomography(H, x, y)
    return p ? ([p[0] * videoWidth, p[1] * videoHeight] as const) : null
  }

  const shuttle = shuttleImageAt(analysis.shuttle, time)
  const trail = shuttleImageTrail(analysis.shuttle, time, TRAIL_SECONDS)
  const players = analysis.players
    .map((track) => {
      const pos = playerPositionAt(track.samples, time)
      const px = pos ? toPx(pos.x, pos.y) : null
      return px ? { id: track.id, px } : null
    })
    .filter((p): p is { id: PlayerId; px: readonly [number, number] } => p !== null)
  const scale = videoHeight / 1080

  return (
    <svg
      className={styles.overlay}
      viewBox={`0 0 ${videoWidth} ${videoHeight}`}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
      data-testid="video-overlay"
    >
      <CourtLines H={H} width={videoWidth} height={videoHeight} />
      {players.map(({ id, px }) => (
        <g key={id} className={id === 'A' ? styles.playerA : styles.playerB}>
          <ellipse cx={px[0]} cy={px[1]} rx={26 * scale} ry={9 * scale} />
          <text x={px[0]} y={px[1] + 30 * scale} fontSize={22 * scale} textAnchor="middle">
            {id}
          </text>
        </g>
      ))}
      {trail.map((run, i) => (
        <polyline
          key={i}
          className={styles.trail}
          strokeWidth={4 * scale}
          points={run.map((p) => `${p.x * videoWidth},${p.y * videoHeight}`).join(' ')}
        />
      ))}
      {shuttle && (
        <circle
          className={styles.shuttle}
          cx={shuttle.x * videoWidth}
          cy={shuttle.y * videoHeight}
          r={11 * scale}
          strokeWidth={3 * scale}
          data-testid="overlay-shuttle"
        />
      )}
    </svg>
  )
}

const CourtLines = memo(function CourtLines({
  H,
  width,
  height,
}: {
  H: readonly number[]
  width: number
  height: number
}) {
  // geometry.ts lines are in the landscape drawing frame (x along, y across);
  // convert to normalized court coordinates (x across, y along) first.
  const segs = courtLines()
    .map(([x1, y1, x2, y2]) => {
      const a = applyHomography(H, y1 / COURT.width, x1 / COURT.length)
      const b = applyHomography(H, y2 / COURT.width, x2 / COURT.length)
      return a && b ? [a[0] * width, a[1] * height, b[0] * width, b[1] * height] : null
    })
    .filter((s): s is number[] => s !== null)
  return (
    <g className={styles.court} strokeWidth={Math.max(2, height / 400)}>
      {segs.map(([x1, y1, x2, y2], i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} />
      ))}
    </g>
  )
})
