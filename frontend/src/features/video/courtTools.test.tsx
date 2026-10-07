import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import type { RallyAnalysis } from '../../types/analysis'
import { CourtMarker } from './CourtMarker'
import { VideoOverlay } from './VideoOverlay'

// Identity-like mapping: normalized court (x, y) -> normalized image (x, 1 - y).
const H = [1, 0, 0, 0, -1, 1, 0, 0, 1]

const ANALYSIS: RallyAnalysis = {
  schemaVersion: 1,
  source: 'cv',
  durationSec: 4,
  players: [{ id: 'A', label: 'Player A', samples: [{ t: 0, position: { x: 0.5, y: 0.25 } }] }],
  shuttle: [
    { t: 1, position: { x: 0.5, y: 0.3 }, image: { x: 0.4, y: 0.6 } },
    { t: 2, position: { x: 0.5, y: 0.6 }, image: { x: 0.6, y: 0.2 } },
    { t: 3, position: { x: 0.5, y: 0.7 }, image: null },
  ],
  shots: [{ index: 0, hitter: 'A', type: 'serve', startTime: 1, endTime: 3 }],
  calibration: { courtToImage: H, confidence: 0.7, manual: false },
}

describe('VideoOverlay', () => {
  function renderAt(time: number) {
    return render(
      <VideoOverlay
        analysis={ANALYSIS}
        calibration={ANALYSIS.calibration!}
        time={time}
        videoWidth={1000}
        videoHeight={500}
      />,
    )
  }

  it('draws every court line through the calibration', () => {
    renderAt(1)
    const svg = screen.getByTestId('video-overlay')
    expect(svg).toHaveAttribute('viewBox', '0 0 1000 500')
    expect(svg.querySelectorAll('line')).toHaveLength(12)
  })

  it('places the detected shuttle at the current time', () => {
    renderAt(1.5)
    const dot = screen.getByTestId('overlay-shuttle')
    expect(Number(dot.getAttribute('cx'))).toBeCloseTo(500)
    expect(Number(dot.getAttribute('cy'))).toBeCloseTo(200)
  })

  it('hides the shuttle when it was not detected', () => {
    renderAt(2.5)
    expect(screen.queryByTestId('overlay-shuttle')).not.toBeInTheDocument()
  })

  it('marks player feet on the floor', () => {
    renderAt(1)
    expect(screen.getByText('A')).toBeInTheDocument()
  })
})

describe('CourtMarker', () => {
  function setup() {
    const onSubmit = vi.fn()
    const onCancel = vi.fn()
    render(
      <CourtMarker videoWidth={1600} videoHeight={900} onSubmit={onSubmit} onCancel={onCancel} />,
    )
    const surface = screen.getByTestId('court-marker-surface')
    // A 400x400 element showing a 16:9 frame: letterboxed to 400x225 at top 87.5.
    vi.spyOn(surface, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 400, height: 400, right: 400, bottom: 400, x: 0, y: 0,
      toJSON: () => ({}),
    })
    return { onSubmit, onCancel, surface }
  }

  it('collects four corners in order, normalized to the video frame', async () => {
    const user = userEvent.setup()
    const { onSubmit, surface } = setup()
    const dialog = screen.getByRole('dialog', { name: 'Mark the court corners' })
    expect(dialog).toHaveTextContent('1 of 4')
    expect(within(dialog).getByRole('button', { name: 'Analyse' })).toBeDisabled()

    const clicks = [
      [40, 300],
      [360, 300],
      [300, 100],
      [100, 100],
    ]
    for (const [x, y] of clicks) fireEvent.click(surface, { clientX: x, clientY: y })
    expect(dialog).toHaveTextContent('All four corners marked')

    await user.click(within(dialog).getByRole('button', { name: 'Analyse' }))
    const corners = onSubmit.mock.calls[0][0]
    expect(corners).toHaveLength(4)
    expect(corners[0].x).toBeCloseTo(0.1)
    expect(corners[0].y).toBeCloseTo((300 - 87.5) / 225)
  })

  it('ignores clicks in the letterbox bars', () => {
    const { surface } = setup()
    fireEvent.click(surface, { clientX: 200, clientY: 20 })
    expect(screen.getByRole('dialog')).toHaveTextContent('1 of 4')
  })

  it('can undo and cancel', async () => {
    const user = userEvent.setup()
    const { onCancel, surface } = setup()
    fireEvent.click(surface, { clientX: 40, clientY: 300 })
    expect(screen.getByRole('dialog')).toHaveTextContent('2 of 4')
    await user.click(screen.getByRole('button', { name: 'Undo' }))
    expect(screen.getByRole('dialog')).toHaveTextContent('1 of 4')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onCancel).toHaveBeenCalled()
  })
})
