import type { Shot, ShotOutcome, ShotType } from '../../types/analysis'

export const SHOT_TYPE_LABEL: Record<ShotType, string> = {
  serve: 'Serve',
  clear: 'Clear',
  drop: 'Drop',
  smash: 'Smash',
  drive: 'Drive',
  net: 'Net shot',
  lift: 'Lift',
  unknown: 'Unknown',
}

export const SHOT_OUTCOME_LABEL: Record<ShotOutcome, string> = {
  net: 'into the net',
  landed: 'landed',
}

/** "Smash", or "Smash · into the net" when that's how the shot ended. */
export function describeShot(shot: Shot): string {
  const type = SHOT_TYPE_LABEL[shot.type]
  return shot.outcome === 'net' ? `${type} · ${SHOT_OUTCOME_LABEL.net}` : type
}
