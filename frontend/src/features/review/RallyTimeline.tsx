import { ChevronLeft, ChevronRight, ListOrdered } from 'lucide-react'
import { useEffect, useMemo, useRef, type KeyboardEvent, type PointerEvent } from 'react'
import { Button } from '../../components/ui/Button'
import { Panel } from '../../components/ui/Panel'
import { formatTime } from '../../lib/format'
import type { RallyAnalysis, Shot } from '../../types/analysis'
import {
  groupRallies,
  nextShotTime,
  previousShotTime,
  shotIndexAt,
  shotPositionLabel,
  shotSeekTime,
  type RallySpan,
} from '../court/sampling'
import { SHOT_TYPE_LABEL } from '../court/shotLabels'
import styles from './RallyTimeline.module.css'

const ARROW_STEP_SEC = 1
const SHIFT_ARROW_STEP_SEC = 5

interface RallyTimelineProps {
  /** Null until analysis is ready; the track still scrubs the video without it. */
  result: RallyAnalysis | null
  /** Video duration in seconds; 0 when no video is loaded. */
  duration: number
  currentTime: number
  onSeek: (time: number) => void
}

/**
 * Full-width rally timeline: shot blocks, a playhead you can drag to scrub the
 * video, prev/next shot navigation and a clickable shot list.
 */
export function RallyTimeline({ result, duration, currentTime, onSeek }: RallyTimelineProps) {
  const shots = useMemo(() => result?.shots ?? [], [result])
  const rallies = useMemo(() => groupRallies(shots), [shots])
  const active = shotIndexAt(shots, currentTime)
  const disabled = duration <= 0
  const prev = previousShotTime(shots, currentTime)
  const next = nextShotTime(shots, currentTime)

  const meta = result ? (
    <div className={styles.nav}>
      <Button
        variant="ghost"
        size="sm"
        icon={<ChevronLeft size={16} />}
        aria-label="Previous shot"
        title="Previous shot"
        disabled={prev === null || disabled}
        onClick={() => prev !== null && onSeek(prev)}
      />
      <span className={styles.navLabel} aria-live="polite">
        {active === null ? summaryLabel(shots.length, rallies.length) : shotPositionLabel(rallies, active)}
      </span>
      <Button
        variant="ghost"
        size="sm"
        icon={<ChevronRight size={16} />}
        aria-label="Next shot"
        title="Next shot"
        disabled={next === null || disabled}
        onClick={() => next !== null && onSeek(next)}
      />
    </div>
  ) : (
    '0 shots'
  )

  return (
    <Panel title="Rally timeline" icon={<ListOrdered size={16} />} meta={meta}>
      <div className={styles.body}>
        <Track
          shots={shots}
          rallies={rallies}
          active={active}
          duration={duration}
          currentTime={currentTime}
          disabled={disabled}
          onSeek={onSeek}
        />
        {result ? (
          <ShotList
            shots={shots}
            rallies={rallies}
            active={active}
            disabled={disabled}
            onSeek={onSeek}
          />
        ) : (
          <p className={styles.text}>Shot markers appear here once a rally has been analysed.</p>
        )}
      </div>
    </Panel>
  )
}

function summaryLabel(shots: number, rallies: number): string {
  const s = `${shots} ${shots === 1 ? 'shot' : 'shots'}`
  return rallies > 1 ? `${rallies} rallies · ${s}` : s
}

interface TrackProps {
  shots: readonly Shot[]
  rallies: readonly RallySpan[]
  active: number | null
  duration: number
  currentTime: number
  disabled: boolean
  onSeek: (time: number) => void
}

function Track({ shots, rallies, active, duration, currentTime, disabled, onSeek }: TrackProps) {
  const trackRef = useRef<HTMLDivElement>(null)
  const pct = (t: number) => (duration > 0 ? `${(Math.min(Math.max(t / duration, 0), 1) * 100).toFixed(3)}%` : '0%')

  const timeFromPointer = (clientX: number) => {
    const rect = trackRef.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return 0
    return (Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1)) * duration
  }

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (disabled || e.button !== 0) return
    e.currentTarget.setPointerCapture?.(e.pointerId)
    e.currentTarget.focus()
    onSeek(timeFromPointer(e.clientX))
  }
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (e.currentTarget.hasPointerCapture?.(e.pointerId)) onSeek(timeFromPointer(e.clientX))
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return
    const step = e.shiftKey ? SHIFT_ARROW_STEP_SEC : ARROW_STEP_SEC
    const targets: Record<string, number | null> = {
      ArrowLeft: currentTime - step,
      ArrowDown: currentTime - step,
      ArrowRight: currentTime + step,
      ArrowUp: currentTime + step,
      Home: 0,
      End: duration,
      PageUp: previousShotTime(shots, currentTime),
      PageDown: nextShotTime(shots, currentTime),
    }
    if (!(e.key in targets)) return
    e.preventDefault()
    const target = targets[e.key]
    if (target !== null) onSeek(Math.min(Math.max(target, 0), duration))
  }

  const activeShot = active !== null ? shots[active] : null
  const valueText =
    `${formatTime(currentTime)} of ${formatTime(duration)}` +
    (activeShot && active !== null
      ? `, shot ${active + 1}: ${SHOT_TYPE_LABEL[activeShot.type]} by Player ${activeShot.hitter}`
      : '')

  return (
    <div className={styles.trackRow}>
      <span className={styles.time}>{formatTime(currentTime)}</span>
      <div
        ref={trackRef}
        className={styles.track}
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label="Rally timeline"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration * 100) / 100}
        aria-valuenow={Math.round(currentTime * 100) / 100}
        aria-valuetext={valueText}
        aria-disabled={disabled || undefined}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onKeyDown={onKeyDown}
        data-testid="timeline-track"
      >
        {shots.map((shot, i) => (
          <span
            key={shot.index}
            className={styles.shotBlock}
            data-hitter={shot.hitter}
            data-active={i === active || undefined}
            style={{ left: pct(shot.startTime), width: `calc(${pct(shot.endTime)} - ${pct(shot.startTime)})` }}
            aria-hidden="true"
          />
        ))}
        {rallies.length > 1 &&
          rallies.map((rally) => (
            <span
              key={rally.index}
              className={styles.rallyMark}
              style={{ left: pct(rally.startTime) }}
              aria-hidden="true"
            />
          ))}
        {!disabled && (
          <>
            <span className={styles.progress} style={{ width: pct(currentTime) }} aria-hidden="true" />
            <span
              className={styles.playhead}
              style={{ left: pct(currentTime) }}
              aria-hidden="true"
              data-testid="playhead"
            />
          </>
        )}
      </div>
      <span className={styles.time}>{formatTime(duration)}</span>
    </div>
  )
}

interface ShotListProps {
  shots: readonly Shot[]
  rallies: readonly RallySpan[]
  active: number | null
  disabled: boolean
  onSeek: (time: number) => void
}

/** Shot chips, one row per rally (labelled when there are several). */
function ShotList({ shots, rallies, active, disabled, onSeek }: ShotListProps) {
  const listRef = useRef<HTMLDivElement>(null)
  const multi = rallies.length > 1

  // Keep the active shot in view inside the list without scrolling the page.
  useEffect(() => {
    const list = listRef.current
    if (!list || active === null) return
    const item = list.querySelector<HTMLElement>('[aria-current="true"]')
    if (!item) return
    const top = item.offsetTop
    const bottom = top + item.offsetHeight
    if (top < list.scrollTop || bottom > list.scrollTop + list.clientHeight) {
      list.scrollTo?.({ top: top - list.clientHeight / 2 + item.offsetHeight / 2, behavior: 'smooth' })
    }
  }, [active])

  return (
    <div ref={listRef} className={styles.rallyList} data-multi={multi || undefined}>
      {rallies.map((rally) => {
        const label = multi ? `Rally ${rally.index + 1}` : 'Shots'
        return (
          <div key={rally.index} className={styles.rallyRow}>
            {multi && (
              <span className={styles.rallyLabel} aria-hidden="true">
                <span className={styles.rallyName}>{label}</span>
                <span className={styles.rallyTime}>{formatTime(rally.startTime)}</span>
              </span>
            )}
            <ol className={styles.shotList} aria-label={label}>
              {shots.slice(rally.first, rally.last + 1).map((shot, k) => {
                const i = rally.first + k
                return (
                  <li key={shot.index}>
                    <button
                      type="button"
                      className={styles.shotChip}
                      data-hitter={shot.hitter}
                      aria-current={i === active ? 'true' : undefined}
                      disabled={disabled}
                      onClick={() => onSeek(shotSeekTime(shot))}
                      title={`Jump to ${multi ? `rally ${rally.index + 1}, ` : ''}shot ${k + 1} (${formatTime(shot.startTime)})`}
                    >
                      <span className={styles.chipNumber}>{k + 1}</span>
                      <span className={styles.chipType}>{SHOT_TYPE_LABEL[shot.type]}</span>
                      <span className={styles.chipHitter} aria-label={`by Player ${shot.hitter}`}>
                        {shot.hitter}
                      </span>
                    </button>
                  </li>
                )
              })}
            </ol>
          </div>
        )
      })}
    </div>
  )
}
