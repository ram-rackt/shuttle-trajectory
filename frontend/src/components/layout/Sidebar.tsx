import { ChartNoAxesColumn, Clapperboard, LibraryBig, Settings, type LucideIcon } from 'lucide-react'
import type { BackendStatus } from '../../hooks/useBackendStatus'
import { LogoMark } from '../ui/Logo'
import styles from './Sidebar.module.css'

interface NavItem {
  label: string
  icon: LucideIcon
  active?: boolean
  /** Placeholder sections that are not built yet. */
  soon?: boolean
}

const PRIMARY_NAV: NavItem[] = [
  { label: 'Review', icon: Clapperboard, active: true },
  { label: 'Library', icon: LibraryBig, soon: true },
  { label: 'Insights', icon: ChartNoAxesColumn, soon: true },
]

const STATUS_LABEL: Record<BackendStatus, string> = {
  checking: 'Connecting…',
  online: 'API online',
  offline: 'API offline',
}

function NavLink({ item }: { item: NavItem }) {
  const Icon = item.icon
  const tooltip = item.soon ? `${item.label} (coming soon)` : item.label
  return (
    <li>
      <a
        href={item.soon ? undefined : '#'}
        className={styles.navItem}
        aria-current={item.active ? 'page' : undefined}
        aria-disabled={item.soon || undefined}
        title={tooltip}
      >
        <Icon size={20} strokeWidth={1.9} aria-hidden="true" />
        <span className={styles.navLabel}>{item.label}</span>
        {item.soon && <span className={styles.soon}>Soon</span>}
      </a>
    </li>
  )
}

export function Sidebar({ backendStatus }: { backendStatus: BackendStatus }) {
  return (
    <aside className={styles.sidebar}>
      <a href="#" className={styles.brand} aria-label="RallyReview home">
        <LogoMark size={32} />
        <span className={styles.wordmark}>
          Rally<span className={styles.wordmarkAccent}>Review</span>
        </span>
      </a>

      <nav className={styles.nav} aria-label="Main">
        <ul className={styles.navList}>
          {PRIMARY_NAV.map((item) => (
            <NavLink key={item.label} item={item} />
          ))}
        </ul>
      </nav>

      <div className={styles.bottom}>
        <ul className={styles.navList}>
          <NavLink item={{ label: 'Settings', icon: Settings, soon: true }} />
        </ul>
        <div
          className={styles.status}
          data-status={backendStatus}
          role="status"
          title={STATUS_LABEL[backendStatus]}
        >
          <span className={styles.statusDot} aria-hidden="true" />
          <span className={styles.navLabel} data-testid="backend-status">
            {STATUS_LABEL[backendStatus]}
          </span>
        </div>
      </div>
    </aside>
  )
}
