import { render, screen } from '@testing-library/react'
import type { RallyAnalysis } from '../../types/analysis'
import { CourtView } from './CourtView'

const RALLY: RallyAnalysis = {
  schemaVersion: 1,
  durationSec: 5,
  players: [
    { id: 'A', label: 'Player A', samples: [{ t: 0, position: { x: 0.5, y: 0.25 } }] },
    { id: 'B', label: 'Player B', samples: [{ t: 0, position: { x: 0.25, y: 0.75 } }] },
  ],
  shuttle: [
    { t: 1, position: { x: 0.5, y: 0.25 } },
    { t: 1.5, position: { x: 0.4, y: 0.5 } },
    { t: 1.75, position: null },
    { t: 2, position: { x: 0.3, y: 0.7 } },
    { t: 2.5, position: { x: 0.3, y: 0.75 } },
    { t: 3.5, position: { x: 0.6, y: 0.3 } },
  ],
  shots: [
    { index: 0, hitter: 'A', type: 'clear', startTime: 1, endTime: 2.5 },
    { index: 1, hitter: 'B', type: 'drop', startTime: 2.5, endTime: 3.5 },
  ],
}

describe('CourtView', () => {
  it('places players using normalized coordinates (y along length, x across width)', () => {
    render(<CourtView analysis={RALLY} time={1} />)
    const a = screen.getByTestId('player-A')
    expect(a.dataset.x).toBe('3.35') // 0.25 × 13.4 m
    expect(a.dataset.y).toBe('3.05') // 0.5 × 6.1 m
    const b = screen.getByTestId('player-B')
    expect(b.dataset.x).toBe('10.05')
    expect(b.dataset.y).toBe('1.52')
  })

  it('draws the shuttle at the given time', () => {
    render(<CourtView analysis={RALLY} time={1.5} />)
    const shuttle = screen.getByTestId('shuttle')
    expect(shuttle.dataset.x).toBe('6.70')
    expect(shuttle.dataset.y).toBe('2.44')
  })

  it('hides the shuttle during a detection gap', () => {
    render(<CourtView analysis={RALLY} time={1.8} />)
    expect(screen.queryByTestId('shuttle')).not.toBeInTheDocument()
  })

  it('highlights the active shot and splits its path at gaps', () => {
    render(<CourtView analysis={RALLY} time={1.2} />)
    const trajectory = screen.getByTestId('trajectory')
    expect(trajectory.querySelectorAll('polyline[data-active]')).toHaveLength(2)
    expect(trajectory.querySelectorAll('polyline[data-shot="1"]:not([data-active])')).toHaveLength(1)
    expect(screen.getByTestId('landing')).toBeInTheDocument()
  })

  it('numbers a marker for every shot and marks the active one', () => {
    render(<CourtView analysis={RALLY} time={3} />)
    const markers = screen.getAllByTestId('shot-marker')
    expect(markers.map((m) => m.textContent)).toEqual(['1', '2'])
    expect(markers[1]).toHaveAttribute('data-active', 'true')
    expect(markers[0]).not.toHaveAttribute('data-active')
  })

  it('describes the current shot for screen readers', () => {
    render(<CourtView analysis={RALLY} time={3} />)
    expect(screen.getByRole('img')).toHaveAccessibleName(
      'Top-down court with 2 shots. Showing shot 2: drop by Player B.',
    )
  })

  it('handles the time before the rally starts', () => {
    render(<CourtView analysis={RALLY} time={0} />)
    expect(screen.getByRole('img')).toHaveAccessibleName(/Rally not started/)
    expect(screen.queryByTestId('landing')).not.toBeInTheDocument()
    expect(screen.getByTestId('player-A')).toBeInTheDocument()
  })
})
