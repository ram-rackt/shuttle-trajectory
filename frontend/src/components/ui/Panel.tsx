import { useId, type ReactNode } from 'react'
import styles from './Panel.module.css'

interface PanelProps {
  title: string
  icon?: ReactNode
  /** Right-aligned content in the header, e.g. status text or a chip. */
  meta?: ReactNode
  footer?: ReactNode
  className?: string
  bodyClassName?: string
  children: ReactNode
}

export function Panel({
  title,
  icon,
  meta,
  footer,
  className,
  bodyClassName,
  children,
}: PanelProps) {
  const titleId = useId()
  return (
    <section
      className={[styles.panel, className].filter(Boolean).join(' ')}
      aria-labelledby={titleId}
    >
      <header className={styles.header}>
        <h2 id={titleId} className={styles.title}>
          {icon && (
            <span className={styles.titleIcon} aria-hidden="true">
              {icon}
            </span>
          )}
          {title}
        </h2>
        {meta && <div className={styles.meta}>{meta}</div>}
      </header>
      <div className={[styles.body, bodyClassName].filter(Boolean).join(' ')}>{children}</div>
      {footer && <footer className={styles.footer}>{footer}</footer>}
    </section>
  )
}
