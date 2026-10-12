/** Formats seconds as m:ss, or h:mm:ss for an hour or more. */
export function formatTime(seconds: number): string {
  const total = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const ss = String(s).padStart(2, '0')
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`
}

/** Formats a byte count as a short human-readable size (e.g. "24.3 MB"). */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / 1024 ** i
  return `${value >= 100 || i === 0 ? Math.round(value) : value.toFixed(1)} ${units[i]}`
}

/** Formats a playback speed, e.g. "0.25×", "1×", "1.5×". */
export function formatRate(rate: number): string {
  return `${rate}×`
}

/**
 * Parses a time typed by a person: "83", "1:23", "1:02:03" or with decimals
 * ("1:23.5"). Returns seconds, or null if it isn't a valid time.
 */
export function parseTimecode(text: string): number | null {
  const parts = text.trim().split(':')
  if (parts.length === 0 || parts.length > 3 || parts.some((p) => !/^\d+(\.\d+)?$/.test(p))) {
    return null
  }
  const nums = parts.map(Number)
  // Minutes and seconds after the first field must be below 60 ("1:75" is a typo).
  if (nums.slice(1).some((n) => n >= 60)) return null
  return nums.reduce((total, n) => total * 60 + n, 0)
}
