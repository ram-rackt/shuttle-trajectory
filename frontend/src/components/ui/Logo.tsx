import styles from './Logo.module.css'

/** RallyReview mark: a tilted shuttlecock on a green tile. */
export function LogoMark({ size = 32 }: { size?: number }) {
  return (
    <span
      className={styles.mark}
      style={{ width: size, height: size, borderRadius: size * 0.19 }}
      aria-hidden="true"
    >
      <svg viewBox="0 0 24 24" width={size * 0.7} height={size * 0.7} fill="none">
        <g transform="rotate(-35 12 12)">
          {/* Feather skirt */}
          <path
            d="M5.5 3.5 Q12 1.8 18.5 3.5 L14 13.8 H10 Z"
            fill="currentColor"
            fillOpacity="0.22"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinejoin="round"
          />
          <path d="M9.8 2.7 11.2 13.6M14.2 2.7l-1.4 10.9" stroke="currentColor" strokeWidth="1.3" />
          {/* Cork */}
          <path d="M9.2 14.6h5.6v1.3a2.8 2.8 0 0 1-5.6 0v-1.3Z" fill="currentColor" />
        </g>
      </svg>
    </span>
  )
}
