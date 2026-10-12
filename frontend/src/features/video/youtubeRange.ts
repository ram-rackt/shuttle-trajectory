import { formatTime } from '../../lib/format'

/** Why a start/end pair can't be analysed, or null if it can. */
export function rangeProblem(
  start: number | null,
  end: number | null,
  duration: number,
  limit: number,
): string | null {
  if (start === null || end === null) return 'Enter times like 1:30 or 1:02:30.'
  if (end <= start) return 'The end must be after the start.'
  if (end - start < 2) return 'Choose at least 2 seconds.'
  if (end > duration + 1) return `The video is only ${formatTime(duration)} long.`
  if (end - start > limit + 1) return `Choose at most ${formatTime(limit)}.`
  return null
}
