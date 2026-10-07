import {
  AlertTriangle,
  Check,
  CircleSlash,
  Crosshair,
  Loader2,
  Play,
  RotateCcw,
} from 'lucide-react'
import type { ReactNode } from 'react'
import { Button } from '../../components/ui/Button'
import type { AnalysisState } from './useAnalysis'
import styles from './AnalysisStatusView.module.css'

interface AnalysisStatusViewProps {
  state: Exclude<AnalysisState, { phase: 'idle' | 'ready' }>
  onStart: () => void
  onCancel: () => void
  /** Offered when the court couldn't be found automatically. */
  onMarkCourt?: () => void
}

/** Failures that marking the court by hand can fix. */
const COURT_ERRORS = new Set(['court_not_found', 'invalid_court_corners'])

/** Court-panel content while there is a video but no court to show yet. */
export function AnalysisStatusView({
  state,
  onStart,
  onCancel,
  onMarkCourt,
}: AnalysisStatusViewProps) {
  switch (state.phase) {
    case 'uploading':
      return (
        <ProgressCard
          title="Uploading video"
          percent={state.progress * 100}
          caption={state.progress >= 1 ? 'Finishing upload…' : 'Sending the rally to the analyser'}
          onCancel={onCancel}
        />
      )
    case 'processing': {
      const { job } = state
      return (
        <ProgressCard
          title="Analysing rally"
          percent={job.progress * 100}
          caption={job.status === 'queued' ? 'Waiting to start…' : undefined}
          onCancel={onCancel}
        >
          <ol className={styles.stages} aria-label="Analysis stages">
            {job.stages.map((label, i) => {
              const current = job.currentStage ?? (job.status === 'queued' ? -1 : job.stages.length)
              const stageState = i < current ? 'done' : i === current ? 'active' : 'pending'
              return (
                <li key={label} className={styles.stage} data-state={stageState}>
                  <span className={styles.stageIcon} aria-hidden="true">
                    {stageState === 'done' && <Check size={12} strokeWidth={3} />}
                    {stageState === 'active' && <Loader2 size={14} className={styles.spin} />}
                  </span>
                  {label}
                  <span className="visually-hidden">
                    {stageState === 'done' ? ' (done)' : stageState === 'active' ? ' (in progress)' : ''}
                  </span>
                </li>
              )
            })}
          </ol>
        </ProgressCard>
      )
    }
    case 'failed': {
      const canMark = Boolean(onMarkCourt && state.jobId && COURT_ERRORS.has(state.code ?? ''))
      return (
        <div className={styles.message} role="alert">
          <span className={`${styles.messageIcon} ${styles.danger}`} aria-hidden="true">
            <AlertTriangle size={22} />
          </span>
          <h3 className={styles.title}>
            {state.during === 'upload' ? 'Upload failed' : 'Analysis failed'}
          </h3>
          <p className={styles.text}>{state.message}</p>
          <div className={styles.buttons}>
            {canMark && (
              <Button variant="primary" icon={<Crosshair size={16} />} onClick={onMarkCourt}>
                Mark court corners
              </Button>
            )}
            <Button
              variant={canMark ? 'secondary' : 'primary'}
              icon={<RotateCcw size={16} />}
              onClick={onStart}
            >
              Try again
            </Button>
          </div>
        </div>
      )
    }
    case 'cancelled':
      return (
        <div className={styles.message}>
          <span className={styles.messageIcon} aria-hidden="true">
            <CircleSlash size={22} />
          </span>
          <h3 className={styles.title}>Analysis cancelled</h3>
          <p className={styles.text}>Start again whenever you’re ready.</p>
          <Button variant="primary" icon={<Play size={16} />} onClick={onStart}>
            Analyse rally
          </Button>
        </div>
      )
  }
}

function ProgressCard({
  title,
  percent,
  caption,
  onCancel,
  children,
}: {
  title: string
  percent: number
  caption?: string
  onCancel: () => void
  children?: ReactNode
}) {
  const rounded = Math.round(Math.min(Math.max(percent, 0), 100))
  return (
    <div className={styles.card} aria-live="polite">
      <div className={styles.cardHeader}>
        <h3 className={styles.title}>{title}</h3>
        <span className={styles.percent}>{rounded}%</span>
      </div>
      <div
        className={styles.bar}
        role="progressbar"
        aria-label={title}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={rounded}
      >
        <div className={styles.barFill} style={{ width: `${rounded}%` }} />
      </div>
      {caption && <p className={styles.caption}>{caption}</p>}
      {children}
      <div className={styles.cardFooter}>
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  )
}

