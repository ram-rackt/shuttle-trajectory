import { Check, Undo2, X } from 'lucide-react'
import { useState, type MouseEvent } from 'react'
import type { CourtCorners } from '../../api/client'
import { Button } from '../../components/ui/Button'
import { containRect } from '../../lib/homography'
import type { ImagePoint } from '../../types/analysis'
import styles from './CourtMarker.module.css'

/** Order the backend expects (near = closest to the camera). */
const CORNER_STEPS = [
  { key: 'near-left', label: 'near-left corner (closest to the camera, on the left)' },
  { key: 'near-right', label: 'near-right corner (closest to the camera, on the right)' },
  { key: 'far-right', label: 'far-right corner (by the far baseline, on the right)' },
  { key: 'far-left', label: 'far-left corner (by the far baseline, on the left)' },
] as const

interface CourtMarkerProps {
  videoWidth: number
  videoHeight: number
  onSubmit: (corners: CourtCorners) => void
  onCancel: () => void
}

/**
 * Lets the user click the four outer court corners on the (paused) video frame
 * when automatic court detection fails. Points are normalized to the video
 * frame, accounting for letterboxing.
 */
export function CourtMarker({ videoWidth, videoHeight, onSubmit, onCancel }: CourtMarkerProps) {
  const [points, setPoints] = useState<ImagePoint[]>([])
  const done = points.length === 4

  const onClick = (e: MouseEvent<HTMLDivElement>) => {
    if (done) return
    const box = containRect(e.currentTarget.getBoundingClientRect(), videoWidth, videoHeight)
    const x = (e.clientX - box.left) / box.width
    const y = (e.clientY - box.top) / box.height
    if (x < 0 || x > 1 || y < 0 || y > 1) return
    setPoints((p) => [...p, { x, y }])
  }

  const step = CORNER_STEPS[Math.min(points.length, 3)]
  return (
    <div className={styles.marker}>
      <div
        className={styles.surface}
        onClick={onClick}
        role="application"
        aria-label="Court corner marking area"
        data-testid="court-marker-surface"
      >
        <svg
          className={styles.svg}
          viewBox={`0 0 ${videoWidth} ${videoHeight}`}
          preserveAspectRatio="xMidYMid meet"
          aria-hidden="true"
        >
          {points.length > 1 && (
            <polygon
              className={done ? styles.quadDone : styles.quad}
              points={points.map((p) => `${p.x * videoWidth},${p.y * videoHeight}`).join(' ')}
              strokeWidth={videoHeight / 300}
            />
          )}
          {points.map((p, i) => (
            <g key={CORNER_STEPS[i].key}>
              <circle
                className={styles.point}
                cx={p.x * videoWidth}
                cy={p.y * videoHeight}
                r={videoHeight / 90}
              />
              <text
                className={styles.pointLabel}
                x={p.x * videoWidth}
                y={p.y * videoHeight - videoHeight / 45}
                fontSize={videoHeight / 40}
                textAnchor="middle"
              >
                {i + 1}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className={styles.panel} role="dialog" aria-label="Mark the court corners">
        <CornerDiagram active={done ? -1 : points.length} />
        <p className={styles.instruction} aria-live="polite">
          {done ? (
            'All four corners marked. Analyse with this court?'
          ) : (
            <>
              <strong>
                {points.length + 1} of 4:
              </strong>{' '}
              click the {step.label} of the outer doubles lines.
            </>
          )}
        </p>
        <div className={styles.actions}>
          <Button
            variant="ghost"
            size="sm"
            icon={<Undo2 size={14} />}
            onClick={() => setPoints((p) => p.slice(0, -1))}
            disabled={points.length === 0}
          >
            Undo
          </Button>
          <Button variant="ghost" size="sm" icon={<X size={14} />} onClick={onCancel}>
            Cancel
          </Button>
          <Button
            variant="primary"
            size="sm"
            icon={<Check size={14} />}
            disabled={!done}
            onClick={() => done && onSubmit(points as CourtCorners)}
          >
            Analyse
          </Button>
        </div>
      </div>
    </div>
  )
}

/** Small court diagram in camera perspective, highlighting the next corner. */
function CornerDiagram({ active }: { active: number }) {
  // near-left, near-right, far-right, far-left in a trapezoid (far end narrower).
  const corners = [
    [6, 42],
    [58, 42],
    [46, 6],
    [18, 6],
  ]
  return (
    <svg className={styles.diagram} viewBox="0 0 64 48" aria-hidden="true">
      <polygon points={corners.map((c) => c.join(',')).join(' ')} />
      <line x1="32" y1="6" x2="32" y2="42" />
      <line x1="12" y1="24" x2="52" y2="24" />
      {corners.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={i === active ? 4.5 : 2.5} data-active={i === active} />
      ))}
    </svg>
  )
}
