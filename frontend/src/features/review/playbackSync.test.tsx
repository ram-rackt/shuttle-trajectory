import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../../App'
import * as api from '../../api/client'
import type { RallyAnalysis } from '../../types/analysis'

vi.mock('../analysis/readVideoDuration', () => ({ readVideoDuration: vi.fn().mockResolvedValue(6) }))
vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof api>()
  const job: api.AnalysisJobStatus = {
    id: 'job-1',
    status: 'ready',
    progress: 1,
    stages: [],
    currentStage: null,
    error: null,
  }
  return {
    ...actual,
    getHealth: vi.fn().mockResolvedValue({ status: 'ok', version: 'test' }),
    createAnalysis: vi.fn().mockResolvedValue(job),
    getAnalysisResult: vi.fn(),
  }
})

const RALLY: RallyAnalysis = {
  schemaVersion: 1,
  durationSec: 6,
  players: [
    {
      id: 'A',
      label: 'Player A',
      samples: [
        { t: 0, position: { x: 0.5, y: 0.25 } },
        { t: 6, position: { x: 0.5, y: 0.25 } },
      ],
    },
    {
      id: 'B',
      label: 'Player B',
      samples: [
        { t: 0, position: { x: 0.5, y: 0.75 } },
        { t: 6, position: { x: 0.5, y: 0.75 } },
      ],
    },
  ],
  shuttle: [
    { t: 1, position: { x: 0.5, y: 0.25 } },
    { t: 2, position: { x: 0.5, y: 0.75 } },
    { t: 3, position: { x: 0.5, y: 0.25 } },
    { t: 4, position: { x: 0.5, y: 0.75 } },
  ],
  shots: [
    { index: 0, hitter: 'A', type: 'serve', startTime: 1, endTime: 2 },
    { index: 1, hitter: 'B', type: 'clear', startTime: 2, endTime: 3 },
    { index: 2, hitter: 'A', type: 'smash', startTime: 3, endTime: 4 },
  ],
}

/** jsdom doesn't play media; give the element a writable clock instead. */
function fakeClock(video: HTMLVideoElement, duration: number) {
  Object.defineProperty(video, 'duration', { configurable: true, value: duration })
  Object.defineProperty(video, 'currentTime', { configurable: true, writable: true, value: 0 })
  fireEvent(video, new Event('loadedmetadata'))
}

function setTime(video: HTMLVideoElement, t: number) {
  act(() => {
    video.currentTime = t
    video.dispatchEvent(new Event('timeupdate'))
  })
}

async function renderReadyRally() {
  vi.mocked(api.getAnalysisResult).mockResolvedValue(RALLY)
  const user = userEvent.setup()
  render(<App />)
  await screen.findByText('API online')
  await user.upload(
    screen.getByLabelText('Choose video file'),
    new File(['x'], 'rally.mp4', { type: 'video/mp4' }),
  )
  const court = screen.getByRole('region', { name: 'Court view' })
  await within(court).findByTestId('court-view')
  const video = screen.getByTestId<HTMLVideoElement>('rally-video')
  fakeClock(video, 6)
  const timeline = screen.getByRole('region', { name: 'Rally timeline' })
  return { user, court, timeline, video }
}

const shotSummary = (court: HTMLElement) => within(court).getByLabelText('Current shot')

describe('playback sync', () => {
  it('moves the court, current shot and playhead with the video clock', async () => {
    const { court, timeline, video } = await renderReadyRally()

    setTime(video, 1.5)
    expect(shotSummary(court)).toHaveTextContent('Shot1 / 3')
    expect(shotSummary(court)).toHaveTextContent('TypeServe')
    // Halfway through the serve the shuttle is at the net (6.7 m).
    expect(within(court).getByTestId('shuttle').dataset.x).toBe('6.70')
    expect(within(timeline).getByTestId('playhead').style.left).toBe('25%')

    setTime(video, 3.25)
    expect(shotSummary(court)).toHaveTextContent('Shot3 / 3')
    expect(shotSummary(court)).toHaveTextContent('HitterPlayer A')
    expect(within(timeline).getByText('Shot 3 / 3')).toBeInTheDocument()
    expect(within(timeline).getByRole('button', { name: /Smash/ })).toHaveAttribute(
      'aria-current',
      'true',
    )
    expect(within(court).getAllByTestId('shuttle-trail').length).toBeGreaterThan(0)
  })

  it('seeks the video when a shot is chosen from the list', async () => {
    const { user, court, timeline, video } = await renderReadyRally()

    await user.click(within(timeline).getByRole('button', { name: /Clear/ }))

    expect(video.currentTime).toBeCloseTo(2)
    expect(shotSummary(court)).toHaveTextContent('Shot2 / 3')
  })

  it('steps between shots with the previous / next buttons', async () => {
    const { user, timeline, video } = await renderReadyRally()
    const next = within(timeline).getByRole('button', { name: 'Next shot' })
    const prev = within(timeline).getByRole('button', { name: 'Previous shot' })

    expect(prev).toBeDisabled()
    await user.click(next)
    expect(video.currentTime).toBeCloseTo(1)
    await user.click(next)
    expect(video.currentTime).toBeCloseTo(2)
    await user.click(prev)
    expect(video.currentTime).toBeCloseTo(1)
  })

  it('scrubs by clicking and dragging on the track', async () => {
    const { timeline, video } = await renderReadyRally()
    const track = within(timeline).getByRole('slider', { name: 'Rally timeline' })
    vi.spyOn(track, 'getBoundingClientRect').mockReturnValue({
      left: 100,
      width: 600,
      top: 0,
      right: 700,
      bottom: 36,
      height: 36,
      x: 100,
      y: 0,
      toJSON: () => ({}),
    })

    fireEvent.pointerDown(track, { clientX: 400, button: 0, pointerId: 1 })
    expect(video.currentTime).toBeCloseTo(3) // halfway along a 6 s video

    track.hasPointerCapture = () => true
    fireEvent.pointerMove(track, { clientX: 250, pointerId: 1 })
    expect(video.currentTime).toBeCloseTo(1.5)

    fireEvent.pointerDown(track, { clientX: 9999, button: 0, pointerId: 1 })
    expect(video.currentTime).toBe(6) // clamped to the end
  })

  it('supports keyboard control of the timeline', async () => {
    const { timeline, video } = await renderReadyRally()
    const track = within(timeline).getByRole('slider', { name: 'Rally timeline' })
    setTime(video, 1.5)

    fireEvent.keyDown(track, { key: 'ArrowRight' })
    expect(video.currentTime).toBeCloseTo(2.5)
    fireEvent.keyDown(track, { key: 'PageDown' })
    expect(video.currentTime).toBeCloseTo(3)
    fireEvent.keyDown(track, { key: 'Home' })
    expect(video.currentTime).toBe(0)
    fireEvent.keyDown(track, { key: 'End' })
    expect(video.currentTime).toBe(6)
    expect(track).toHaveAttribute('aria-valuetext', expect.stringContaining('shot 3: Smash'))
  })
})
