import { Check } from 'lucide-react'
import styles from './ReviewSteps.module.css'

export type ReviewStep = 'upload' | 'process' | 'review'

const STEPS: { id: ReviewStep; label: string }[] = [
  { id: 'upload', label: 'Upload' },
  { id: 'process', label: 'Analyse' },
  { id: 'review', label: 'Review' },
]

export function ReviewSteps({ current }: { current: ReviewStep }) {
  const currentIndex = STEPS.findIndex((s) => s.id === current)
  return (
    <ol className={styles.steps} aria-label="Analysis progress">
      {STEPS.map((step, i) => {
        const state = i < currentIndex ? 'done' : i === currentIndex ? 'current' : 'todo'
        return (
          <li
            key={step.id}
            className={styles.step}
            data-state={state}
            aria-current={state === 'current' ? 'step' : undefined}
          >
            <span className={styles.badge} aria-hidden="true">
              {state === 'done' ? <Check size={12} strokeWidth={3} /> : i + 1}
            </span>
            <span className={styles.label}>{step.label}</span>
          </li>
        )
      })}
    </ol>
  )
}
