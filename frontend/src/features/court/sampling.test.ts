import type { RallyAnalysis } from '../../types/analysis'
import {
  buildShotPaths,
  nextShotTime,
  playerPositionAt,
  previousShotTime,
  shotIndexAt,
  shuttlePositionAt,
  shuttleTrail,
} from './sampling'

const RALLY: RallyAnalysis = {
  schemaVersion: 1,
  durationSec: 4,
  players: [
    {
      id: 'A',
      label: 'Player A',
      samples: [
        { t: 0, position: { x: 0.4, y: 0.2 } },
        { t: 2, position: { x: 0.6, y: 0.3 } },
      ],
    },
    { id: 'B', label: 'Player B', samples: [{ t: 0, position: { x: 0.5, y: 0.8 } }] },
  ],
  shuttle: [
    { t: 1, position: { x: 0.4, y: 0.2 } },
    { t: 1.5, position: { x: 0.5, y: 0.5 } },
    { t: 2, position: null },
    { t: 2.5, position: { x: 0.5, y: 0.7 } },
    { t: 3, position: { x: 0.6, y: 0.8 } },
    { t: 3.5, position: { x: 0.5, y: 0.6 } },
  ],
  shots: [
    { index: 0, hitter: 'A', type: 'serve', startTime: 1, endTime: 3 },
    { index: 1, hitter: 'B', type: 'drop', startTime: 3, endTime: 3.5 },
  ],
}

describe('playerPositionAt', () => {
  const samples = RALLY.players[0].samples

  it('interpolates between samples', () => {
    expect(playerPositionAt(samples, 1)).toEqual({ x: 0.5, y: 0.25 })
  })

  it('clamps before the first and after the last sample', () => {
    expect(playerPositionAt(samples, -5)).toEqual({ x: 0.4, y: 0.2 })
    expect(playerPositionAt(samples, 99)).toEqual({ x: 0.6, y: 0.3 })
  })

  it('returns null for an empty track', () => {
    expect(playerPositionAt([], 1)).toBeNull()
  })
})

describe('shuttlePositionAt', () => {
  it('is null before tracking starts', () => {
    expect(shuttlePositionAt(RALLY.shuttle, 0.5)).toBeNull()
  })

  it('interpolates between tracked samples', () => {
    const p = shuttlePositionAt(RALLY.shuttle, 1.25)!
    expect(p.x).toBeCloseTo(0.45)
    expect(p.y).toBeCloseTo(0.35)
  })

  it('is null inside a detection gap rather than inventing a position', () => {
    expect(shuttlePositionAt(RALLY.shuttle, 1.75)).toBeNull()
    expect(shuttlePositionAt(RALLY.shuttle, 2)).toBeNull()
    expect(shuttlePositionAt(RALLY.shuttle, 2.25)).toBeNull()
  })

  it('returns exact samples and holds the last one', () => {
    expect(shuttlePositionAt(RALLY.shuttle, 2.5)).toEqual({ x: 0.5, y: 0.7 })
    expect(shuttlePositionAt(RALLY.shuttle, 10)).toEqual({ x: 0.5, y: 0.6 })
  })
})

describe('shotIndexAt', () => {
  it.each([
    [0.5, null],
    [1, 0],
    [2.9, 0],
    [3, 1],
    [10, 1],
  ])('at %s s is %s', (t, expected) => {
    expect(shotIndexAt(RALLY.shots, t)).toBe(expected)
  })

  it('handles no shots', () => {
    expect(shotIndexAt([], 1)).toBeNull()
  })
})

describe('buildShotPaths', () => {
  const paths = buildShotPaths(RALLY)

  it('splits a shot into segments at detection gaps', () => {
    expect(paths[0].segments).toEqual([
      [
        { x: 0.4, y: 0.2 },
        { x: 0.5, y: 0.5 },
      ],
      [
        { x: 0.5, y: 0.7 },
        { x: 0.6, y: 0.8 },
      ],
    ])
  })

  it('records hit and end points', () => {
    expect(paths[0].hitPoint).toEqual({ x: 0.4, y: 0.2 })
    expect(paths[0].endPoint).toEqual({ x: 0.6, y: 0.8 })
    expect(paths[1].hitPoint).toEqual({ x: 0.6, y: 0.8 })
    expect(paths[1].endPoint).toEqual({ x: 0.5, y: 0.6 })
  })

  it("falls back to the hitter's position when the hit wasn't tracked", () => {
    const rally: RallyAnalysis = {
      ...RALLY,
      shuttle: RALLY.shuttle.map((s) => (s.t === 1 ? { ...s, position: null } : s)),
    }
    expect(buildShotPaths(rally)[0].hitPoint).toEqual({ x: 0.5, y: 0.25 })
  })
})

describe('shuttleTrail', () => {
  it('returns recent tracked positions ending at the current position', () => {
    const trail = shuttleTrail(RALLY.shuttle, 1.25, 0.5)
    expect(trail).toHaveLength(1)
    expect(trail[0][0]).toEqual({ x: 0.4, y: 0.2 })
    expect(trail[0].at(-1)!.x).toBeCloseTo(0.45)
  })

  it('splits at gaps and drops single-point runs', () => {
    // Window covers 1.5 (lone tracked point, dropped), 2.0 (gap), 2.5 and 3.0.
    const trail = shuttleTrail(RALLY.shuttle, 3, 1.5)
    expect(trail).toEqual([
      [
        { x: 0.5, y: 0.7 },
        { x: 0.6, y: 0.8 },
        { x: 0.6, y: 0.8 },
      ],
    ])
  })

  it('is empty before tracking starts', () => {
    expect(shuttleTrail(RALLY.shuttle, 0.5, 1)).toEqual([])
  })
})

describe('shot navigation', () => {
  it('finds the next shot start', () => {
    expect(nextShotTime(RALLY.shots, 0)).toBeCloseTo(1)
    expect(nextShotTime(RALLY.shots, 1.2)).toBeCloseTo(3)
    expect(nextShotTime(RALLY.shots, 3.2)).toBeNull()
  })

  it('does not get stuck on the shot it just seeked to', () => {
    const t = nextShotTime(RALLY.shots, 0)!
    expect(nextShotTime(RALLY.shots, t)).toBeCloseTo(3)
  })

  it('restarts the current shot when well into it, else goes back one', () => {
    expect(previousShotTime(RALLY.shots, 3.8)).toBeCloseTo(3)
    expect(previousShotTime(RALLY.shots, 3.1)).toBeCloseTo(1)
    expect(previousShotTime(RALLY.shots, 1.1)).toBeCloseTo(1)
    expect(previousShotTime(RALLY.shots, 0.5)).toBeNull()
  })
})
