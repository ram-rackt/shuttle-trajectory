import { Crosshair, FileJson, Route } from 'lucide-react'
import { analysisResultUrl } from '../../api/client'
import { Chip } from '../../components/ui/Chip'
import { Panel } from '../../components/ui/Panel'
import type { RallyAnalysis } from '../../types/analysis'
import { AnalysisStatusView } from '../analysis/AnalysisStatusView'
import type { AnalysisState } from '../analysis/useAnalysis'
import styles from './CourtPanel.module.css'
import { CourtView } from './CourtView'
import { shotIndexAt } from './sampling'
import { SHOT_TYPE_LABEL } from './shotLabels'

interface CourtPanelProps {
  analysis: AnalysisState
  /** Seconds into the video; the court is drawn for this instant. */
  time: number
  onStartAnalysis: () => void
  onCancelAnalysis: () => void
  /** Starts manual court marking on the video (when the court fit is wrong or missing). */
  onMarkCourt?: () => void
}

/** Right panel: the analysis lifecycle, then the top-down court once ready. */
export function CourtPanel({
  analysis,
  time,
  onStartAnalysis,
  onCancelAnalysis,
  onMarkCourt,
}: CourtPanelProps) {
  const ready = analysis.phase === 'ready' ? analysis : null
  const displayTime = ready ? time : 0
  const simulated = ready?.result.source !== 'cv'

  return (
    <Panel
      title="Court view"
      icon={<Route size={16} />}
      meta={
        <>
          {ready && !simulated && onMarkCourt && (
            <button
              type="button"
              className={styles.rawLink}
              onClick={onMarkCourt}
              title="Court lines don't match the video? Mark the corners by hand."
            >
              <Crosshair size={14} aria-hidden="true" />
              Adjust court
            </button>
          )}
          {ready && (
            <a
              className={styles.rawLink}
              href={analysisResultUrl(ready.job.id)}
              target="_blank"
              rel="noreferrer"
              title="Open the raw analysis JSON"
            >
              <FileJson size={14} aria-hidden="true" />
              Data
            </a>
          )}
          {ready && simulated ? (
            <Chip>Simulated data</Chip>
          ) : (
            <Chip tone={ready ? 'accent' : 'neutral'}>2D · Top-down</Chip>
          )}
        </>
      }
      footer={<ShotSummary result={ready?.result ?? null} time={displayTime} />}
    >
      <div className={styles.stage} data-ready={ready ? true : undefined}>
        {analysis.phase === 'ready' ? (
          <div className={styles.courtWrap}>
            <CourtView analysis={analysis.result} time={displayTime} />
            <Legend />
          </div>
        ) : analysis.phase === 'idle' ? (
          <div className={styles.empty}>
            <CourtGlyph />
            <h3 className={styles.emptyTitle}>No analysis yet</h3>
            <p className={styles.emptyText}>
              Upload a rally to see both players, the shuttle and its trajectory on a top-down
              court, synced to the video.
            </p>
          </div>
        ) : (
          <AnalysisStatusView
            state={analysis}
            onStart={onStartAnalysis}
            onCancel={onCancelAnalysis}
            onMarkCourt={onMarkCourt}
          />
        )}
      </div>
    </Panel>
  )
}

function Legend() {
  return (
    <ul className={styles.legend} aria-label="Legend">
      <li>
        <span className={`${styles.swatch} ${styles.swatchA}`} aria-hidden="true" />
        Player A
      </li>
      <li>
        <span className={`${styles.swatch} ${styles.swatchB}`} aria-hidden="true" />
        Player B
      </li>
      <li>
        <span className={`${styles.swatch} ${styles.swatchShuttle}`} aria-hidden="true" />
        Shuttle
      </li>
      <li>
        <span className={styles.lineSolid} aria-hidden="true" />
        Current shot
      </li>
      <li>
        <span className={styles.lineDashed} aria-hidden="true" />
        Other shots
      </li>
    </ul>
  )
}

function ShotSummary({ result, time }: { result: RallyAnalysis | null; time: number }) {
  const index = result ? shotIndexAt(result.shots, time) : null
  const shot = result && index !== null ? result.shots[index] : null
  const items = [
    { label: 'Shot', value: shot && result ? `${index! + 1} / ${result.shots.length}` : '—' },
    { label: 'Type', value: shot ? SHOT_TYPE_LABEL[shot.type] : '—' },
    { label: 'Hitter', value: shot ? `Player ${shot.hitter}` : '—' },
  ]
  return (
    <dl className={styles.summary} aria-label="Current shot">
      {items.map((item) => (
        <div key={item.label} className={styles.summaryItem}>
          <dt>{item.label}</dt>
          <dd data-filled={item.value !== '—' || undefined}>{item.value}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Small decorative court outline for the empty state (not the real renderer). */
function CourtGlyph() {
  return (
    <svg
      className={styles.glyph}
      viewBox="0 0 120 56"
      width="120"
      height="56"
      fill="none"
      aria-hidden="true"
    >
      <rect x="1" y="1" width="118" height="54" rx="3" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M60 1v54M1 5h118M1 51h118M42 1v54M78 1v54M8 5v46M112 5v46M8 28h34M78 28h34"
        stroke="currentColor"
        strokeWidth="1"
        opacity="0.6"
      />
      <path
        d="M20 40 C 38 10, 70 6, 98 18"
        stroke="var(--accent)"
        strokeWidth="1.6"
        strokeDasharray="3 3"
        strokeLinecap="round"
      />
      <circle cx="98" cy="18" r="3" fill="var(--accent)" />
    </svg>
  )
}
