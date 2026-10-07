import type { ReactNode } from 'react'
import styles from './Chip.module.css'

type Tone = 'neutral' | 'accent'

export function Chip({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`${styles.chip} ${styles[tone]}`}>{children}</span>
}
