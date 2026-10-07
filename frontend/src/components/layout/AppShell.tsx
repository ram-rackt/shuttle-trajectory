import type { ReactNode } from 'react'
import styles from './AppShell.module.css'

interface AppShellProps {
  sidebar: ReactNode
  header: ReactNode
  children: ReactNode
}

export function AppShell({ sidebar, header, children }: AppShellProps) {
  return (
    <div className={styles.shell}>
      {sidebar}
      <div className={styles.content}>
        {header}
        <main className={styles.main}>{children}</main>
      </div>
    </div>
  )
}
