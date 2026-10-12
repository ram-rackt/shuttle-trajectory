import { memo, useMemo } from 'react'
import type { CourtPoint, PlayerId, RallyAnalysis, Shot } from '../../types/analysis'
import { COURT, courtLines, NET_X, toPolylinePoints, toSvg } from './geometry'
import {
  buildShotPaths,
  groupRallies,
  playerPositionAt,
  rallyOfShot,
  shotIndexAt,
  shuttlePositionAt,
  shuttleTrail,
  type RallySpan,
  type ShotPath,
} from './sampling'
import { SHOT_TYPE_LABEL } from './shotLabels'
import styles from './CourtView.module.css'

const PAD_X = 0.9
const PAD_TOP = 0.8
const PAD_BOTTOM = 0.45
const VIEW_BOX = [
  -PAD_X,
  -PAD_TOP,
  COURT.length + PAD_X * 2,
  COURT.width + PAD_TOP + PAD_BOTTOM,
].join(' ')

const NO_PATHS: ShotPath[] = []

/** Seconds of shuttle history drawn behind the shuttle. */
const TRAIL_SECONDS = 0.35

const PLAYER_CLASS: Record<PlayerId, string> = { A: styles.playerA, B: styles.playerB }

interface CourtViewProps {
  analysis: RallyAnalysis
  /** Seconds into the video. Everything drawn is derived from this instant. */
  time: number
}

/**
 * Top-down 2D court at a given time. A pure function of (analysis, time);
 * static layers are memoized so only players and the shuttle re-render as
 * time changes. When the clip holds several rallies, only the current one's
 * shots are drawn (numbered within that rally).
 */
export function CourtView({ analysis, time }: CourtViewProps) {
  // Per rally, the paths of its shots. Built once so each list keeps its identity
  // and the memoized trajectory layer only re-renders when the rally changes.
  const { rallies, pathsByRally } = useMemo(() => {
    const paths = buildShotPaths(analysis)
    const rallies = groupRallies(analysis.shots)
    return { rallies, pathsByRally: rallies.map((r) => paths.slice(r.first, r.last + 1)) }
  }, [analysis])
  const activeShot = shotIndexAt(analysis.shots, time)
  const rally = (activeShot !== null ? rallyOfShot(rallies, activeShot) : null) ?? rallies[0] ?? null
  const rallyPaths = rally ? pathsByRally[rally.index] : NO_PATHS
  const shuttle = shuttlePositionAt(analysis.shuttle, time)
  const trail = shuttleTrail(analysis.shuttle, time, TRAIL_SECONDS)
  const players = analysis.players.map((track) => ({
    id: track.id,
    position: playerPositionAt(track.samples, time),
  }))

  return (
    <svg
      className={styles.court}
      viewBox={VIEW_BOX}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={describe(analysis.shots, rallies, activeShot)}
      data-testid="court-view"
    >
      <CourtSurface />
      <TrajectoryLayer
        paths={rallyPaths}
        shots={analysis.shots}
        activeShot={activeShot}
        firstShot={rally?.first ?? 0}
      />
      {players.map(
        ({ id, position }) => position && <PlayerMarker key={id} id={id} position={position} />,
      )}
      {trail.map((segment, i) => (
        <polyline
          key={i}
          className={styles.trail}
          points={toPolylinePoints(segment)}
          aria-hidden="true"
          data-testid="shuttle-trail"
        />
      ))}
      {shuttle && <ShuttleMarker position={shuttle} />}
    </svg>
  )
}

function describe(shots: readonly Shot[], rallies: readonly RallySpan[], active: number | null): string {
  const base =
    rallies.length > 1
      ? `Top-down court with ${rallies.length} rallies, ${shots.length} shots.`
      : `Top-down court with ${shots.length} shots.`
  if (active === null) return `${base} Rally not started.`
  const shot = shots[active]
  const rally = rallyOfShot(rallies, active)
  const number = rally ? active - rally.first + 1 : active + 1
  const where = rallies.length > 1 && rally ? `rally ${rally.index + 1}, shot ${number}` : `shot ${number}`
  return `${base} Showing ${where}: ${SHOT_TYPE_LABEL[shot.type].toLowerCase()} by Player ${shot.hitter}.`
}

const CourtSurface = memo(function CourtSurface() {
  const { length: L, width: W } = COURT
  return (
    <g aria-hidden="true">
      <rect
        className={styles.surround}
        x={-PAD_X}
        y={-PAD_TOP}
        width={L + PAD_X * 2}
        height={W + PAD_TOP + PAD_BOTTOM}
        rx={0.25}
      />
      <rect className={styles.surface} x={0} y={0} width={L} height={W} />
      {courtLines().map(([x1, y1, x2, y2], i) => (
        <line
          key={i}
          className={styles.line}
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          strokeWidth={COURT.lineWidth * 1.5}
        />
      ))}
      {/* Net and posts, slightly outside the doubles sidelines */}
      <line className={styles.netShadow} x1={NET_X + 0.06} y1={-0.2} x2={NET_X + 0.06} y2={W + 0.2} />
      <line className={styles.net} x1={NET_X} y1={-0.2} x2={NET_X} y2={W + 0.2} />
      <circle className={styles.post} cx={NET_X} cy={-0.2} r={0.07} />
      <circle className={styles.post} cx={NET_X} cy={W + 0.2} r={0.07} />
      <text className={`${styles.sideLabel} ${styles.playerAText}`} x={0} y={-0.32}>
        PLAYER A
      </text>
      <text
        className={`${styles.sideLabel} ${styles.playerBText}`}
        x={L}
        y={-0.32}
        textAnchor="end"
      >
        PLAYER B
      </text>
      <text className={styles.netLabel} x={NET_X} y={-0.32} textAnchor="middle">
        NET
      </text>
    </g>
  )
})

interface TrajectoryLayerProps {
  /** Shots of the rally being shown. */
  paths: ShotPath[]
  shots: readonly Shot[]
  activeShot: number | null
  /** Index of the rally's first shot, for numbering within the rally. */
  firstShot: number
}

const TrajectoryLayer = memo(function TrajectoryLayer({
  paths,
  shots,
  activeShot,
  firstShot,
}: TrajectoryLayerProps) {
  const active = paths.find((p) => p.shotIndex === activeShot) ?? null
  return (
    <g aria-hidden="true" data-testid="trajectory">
      {/* Other shots, faint, underneath */}
      {paths.map(
        (path) =>
          path.shotIndex !== activeShot &&
          path.segments.map((segment, i) => (
            <polyline
              key={`${path.shotIndex}-${i}`}
              className={styles.pathFaint}
              points={toPolylinePoints(segment)}
              data-shot={path.shotIndex}
            />
          )),
      )}

      {/* Active shot on top, with its landing point */}
      {active?.segments.map((segment, i) => (
        <polyline
          key={`active-${i}`}
          className={styles.pathActive}
          points={toPolylinePoints(segment)}
          data-shot={active.shotIndex}
          data-active="true"
        />
      ))}
      {active?.endPoint && <LandingRing point={active.endPoint} />}

      {/* Numbered markers where each shot was struck; the active one drawn last, on top */}
      {[...paths.filter((p) => p !== active), ...(active ? [active] : [])].map(
        (path) =>
          path.hitPoint && (
            <ShotMarker
              key={path.shotIndex}
              point={path.hitPoint}
              number={path.shotIndex - firstShot + 1}
              hitter={shots[path.shotIndex].hitter}
              active={path.shotIndex === activeShot}
            />
          ),
      )}
    </g>
  )
})

function LandingRing({ point }: { point: CourtPoint }) {
  const { x, y } = toSvg(point)
  return (
    <g className={styles.landing} data-testid="landing">
      <circle cx={x} cy={y} r={0.22} />
      <path d={`M${x - 0.12} ${y - 0.12}L${x + 0.12} ${y + 0.12}M${x + 0.12} ${y - 0.12}L${x - 0.12} ${y + 0.12}`} />
    </g>
  )
}

function ShotMarker({
  point,
  number,
  hitter,
  active,
}: {
  point: CourtPoint
  number: number
  hitter: PlayerId
  active: boolean
}) {
  const { x, y } = toSvg(point)
  return (
    <g
      className={`${styles.shotMarker} ${PLAYER_CLASS[hitter]}`}
      data-active={active || undefined}
      data-testid="shot-marker"
    >
      <circle cx={x} cy={y} r={active ? 0.2 : 0.16} />
      <text x={x} y={y} dy="0.07" textAnchor="middle">
        {number}
      </text>
    </g>
  )
}

function PlayerMarker({ id, position }: { id: PlayerId; position: CourtPoint }) {
  const { x, y } = toSvg(position)
  return (
    <g
      className={`${styles.player} ${PLAYER_CLASS[id]}`}
      data-testid={`player-${id}`}
      data-x={x.toFixed(2)}
      data-y={y.toFixed(2)}
    >
      <circle className={styles.playerHalo} cx={x} cy={y} r={0.46} />
      <circle className={styles.playerBody} cx={x} cy={y} r={0.3} />
      <text x={x} y={y} dy="0.11" textAnchor="middle">
        {id}
      </text>
    </g>
  )
}

function ShuttleMarker({ position }: { position: CourtPoint }) {
  const { x, y } = toSvg(position)
  return (
    <g
      className={styles.shuttle}
      data-testid="shuttle"
      data-x={x.toFixed(2)}
      data-y={y.toFixed(2)}
    >
      <circle className={styles.shuttleGlow} cx={x} cy={y} r={0.24} />
      <circle className={styles.shuttleBody} cx={x} cy={y} r={0.1} />
    </g>
  )
}
