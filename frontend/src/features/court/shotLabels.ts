import type { ShotType } from '../../types/analysis'

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
