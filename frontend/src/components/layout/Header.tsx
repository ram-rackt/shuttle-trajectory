import type { ReactNode } from 'react'
import { LogoMark } from '../ui/Logo'
import styles from './Header.module.css'

interface HeaderProps {
  title: string
  subtitle?: string
  /** Centre slot, e.g. the workflow stepper. Hidden on small screens. */
  center?: ReactNode
  actions?: ReactNode
}

export function Header({ title, subtitle, center, actions }: HeaderProps) {
  return (
    <header className={styles.header}>
      <div className={styles.titleGroup}>
        <span className={styles.mobileMark}>
          <LogoMark size={28} />
        </span>
        <div className={styles.titles}>
          <h1 className={styles.title}>{title}</h1>
          {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
        </div>
      </div>
      {center && <div className={styles.center}>{center}</div>}
      {actions && <div className={styles.actions}>{actions}</div>}
    </header>
  )
}
