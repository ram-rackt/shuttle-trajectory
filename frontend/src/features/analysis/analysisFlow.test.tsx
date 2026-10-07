import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, vi } from 'vitest'
import App from '../../App'
import * as api from '../../api/client'
import type { RallyAnalysis } from '../../types/analysis'

vi.mock('./readVideoDuration', () => ({ readVideoDuration: vi.fn().mockResolvedValue(12.5) }))
vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof api>()
  return {
    ...actual,
    getHealth: vi.fn().mockResolvedValue({ status: 'ok', version: 'test' }),
    createAnalysis: vi.fn(),
    getAnalysisStatus: vi.fn(),
    getAnalysisResult: vi.fn(),
  }
})

const STAGES = ['Detecting court', 'Tracking players', 'Tracking shuttle']

function job(overrides: Partial<api.AnalysisJobStatus> = {}): api.AnalysisJobStatus {
  return {
    id: 'job-1',
    status: 'queued',
    progress: 0,
    stages: STAGES,
    currentStage: null,
    error: null,
    ...overrides,
  }
}

const RESULT: RallyAnalysis = {
  schemaVersion: 1,
  durationSec: 12.5,
  players: [],
  shuttle: [
    { t: 1, position: { x: 0.5, y: 0.2 } },
    { t: 1.1, position: null },
  ],
  shots: [
    { index: 0, hitter: 'A', type: 'serve', startTime: 1, endTime: 2.2 },
    { index: 1, hitter: 'B', type: 'clear', startTime: 2.2, endTime: 3.8 },
  ],
}

const createAnalysis = vi.mocked(api.createAnalysis)
const getAnalysisStatus = vi.mocked(api.getAnalysisStatus)
const getAnalysisResult = vi.mocked(api.getAnalysisResult)

beforeEach(() => {
  createAnalysis.mockReset()
  getAnalysisStatus.mockReset()
  getAnalysisResult.mockReset()
})

afterEach(() => {
  vi.restoreAllMocks()
})

async function uploadVideo() {
  const user = userEvent.setup()
  render(<App />)
  await screen.findByText('API online')
  await user.upload(
    screen.getByLabelText('Choose video file'),
    new File(['x'], 'final.mp4', { type: 'video/mp4' }),
  )
  return { user, court: screen.getByRole('region', { name: 'Court view' }) }
}

describe('analysis flow', () => {
  it('goes Upload → Processing → Ready and draws the court', async () => {
    createAnalysis.mockImplementation(async (_file, _duration, opts) => {
      opts?.onUploadProgress?.(0.5)
      return job()
    })
    getAnalysisStatus
      .mockResolvedValueOnce(job({ status: 'processing', progress: 0.4, currentStage: 1 }))
      .mockResolvedValueOnce(job({ status: 'ready', progress: 1 }))
    getAnalysisResult.mockResolvedValue(RESULT)

    const { court } = await uploadVideo()
    expect(createAnalysis).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'final.mp4' }),
      12.5,
      expect.any(Object),
    )

    // Processing: stage checklist reflects the current stage.
    const stages = await within(court).findByRole('list', { name: 'Analysis stages' })
    await waitFor(() =>
      expect(within(stages).getByText('Tracking players').closest('li')).toHaveAttribute(
        'data-state',
        'active',
      ),
    )
    expect(within(stages).getByText('Detecting court').closest('li')).toHaveAttribute(
      'data-state',
      'done',
    )
    const steps = screen.getByRole('list', { name: 'Analysis progress' })
    expect(within(steps).getByText('Analyse').closest('li')).toHaveAttribute('aria-current', 'step')

    // Ready.
    expect(await within(court).findByTestId('court-view', {}, { timeout: 3000 })).toBeVisible()
    const summary = within(court).getByLabelText('Current shot')
    // Court follows the video clock; at 0:00 the rally hasn't started yet.
    expect(summary).toHaveTextContent('Shot—')
    expect(within(steps).getByText('Review').closest('li')).toHaveAttribute('aria-current', 'step')
    expect(screen.getByRole('heading', { level: 1, name: 'Rally review' })).toBeInTheDocument()
    expect(screen.getByText('final.mp4 · 2 shots')).toBeInTheDocument()
  })

  it('shows a retryable error when the upload fails', async () => {
    createAnalysis.mockRejectedValueOnce(new api.ApiError(415, 'Unsupported video format.'))
    const { user, court } = await uploadVideo()

    const alert = await within(court).findByRole('alert')
    expect(alert).toHaveTextContent('Upload failed')
    expect(alert).toHaveTextContent('Unsupported video format.')

    createAnalysis.mockResolvedValueOnce(job({ status: 'ready', progress: 1 }))
    getAnalysisResult.mockResolvedValue(RESULT)
    await user.click(within(court).getByRole('button', { name: 'Try again' }))
    expect(await within(court).findByTestId('court-view')).toBeVisible()
    expect(createAnalysis).toHaveBeenCalledTimes(2)
  })

  it('explains an unreachable backend', async () => {
    createAnalysis.mockRejectedValueOnce(new TypeError('Network error during upload'))
    const { court } = await uploadVideo()
    expect(await within(court).findByRole('alert')).toHaveTextContent('Couldn’t reach')
  })

  it('reports a failed job from the server', async () => {
    createAnalysis.mockResolvedValue(job())
    getAnalysisStatus.mockResolvedValue(job({ status: 'failed', error: 'Analysis failed.' }))
    const { court } = await uploadVideo()
    const alert = await within(court).findByRole('alert', {}, { timeout: 3000 })
    expect(alert).toHaveTextContent('Analysis failed.')
  })

  it('can be cancelled and restarted', async () => {
    createAnalysis.mockReturnValue(new Promise(() => {})) // never finishes
    const { user, court } = await uploadVideo()

    await user.click(await within(court).findByRole('button', { name: 'Cancel' }))
    expect(within(court).getByText('Analysis cancelled')).toBeVisible()

    await user.click(within(court).getByRole('button', { name: 'Analyse rally' }))
    expect(await within(court).findByText('Uploading video')).toBeVisible()
    expect(createAnalysis).toHaveBeenCalledTimes(2)
  })

  it('resets when the video is removed', async () => {
    createAnalysis.mockResolvedValue(job({ status: 'ready', progress: 1 }))
    getAnalysisResult.mockResolvedValue(RESULT)
    const { user, court } = await uploadVideo()
    await within(court).findByTestId('court-view')

    await user.click(screen.getByRole('button', { name: 'Remove video' }))
    expect(within(court).getByText('No analysis yet')).toBeVisible()
    expect(screen.getByRole('region', { name: 'Rally timeline' })).toHaveTextContent('0 shots')
  })
})

describe('manual court calibration', () => {
  it('offers court marking when the court is not found, then re-runs with the corners', async () => {
    createAnalysis.mockResolvedValue(job())
    getAnalysisStatus.mockResolvedValue(
      job({ status: 'failed', error: 'Mark the court corners.', errorCode: 'court_not_found' }),
    )
    const recalibrate = vi
      .spyOn(api, 'recalibrateAnalysis')
      .mockResolvedValue(job({ id: 'job-2', status: 'ready', progress: 1 }))
    getAnalysisResult.mockResolvedValue({ ...RESULT, source: 'cv' })

    const { user, court } = await uploadVideo()
    const video = screen.getByTestId<HTMLVideoElement>('rally-video')
    Object.defineProperty(video, 'videoWidth', { configurable: true, value: 1600 })
    Object.defineProperty(video, 'videoHeight', { configurable: true, value: 900 })
    video.dispatchEvent(new Event('loadedmetadata'))

    await user.click(
      await within(court).findByRole('button', { name: 'Mark court corners' }, { timeout: 3000 }),
    )
    const surface = await screen.findByTestId('court-marker-surface')
    vi.spyOn(surface, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, width: 1600, height: 900, right: 1600, bottom: 900, x: 0, y: 0,
      toJSON: () => ({}),
    })
    for (const [x, y] of [[300, 800], [1300, 800], [1100, 300], [500, 300]]) {
      fireEvent.click(surface, { clientX: x, clientY: y })
    }
    await user.click(screen.getByRole('button', { name: 'Analyse' }))

    expect(recalibrate).toHaveBeenCalledWith(
      'job-1',
      [
        { x: 300 / 1600, y: 800 / 900 },
        { x: 1300 / 1600, y: 800 / 900 },
        { x: 1100 / 1600, y: 300 / 900 },
        { x: 500 / 1600, y: 300 / 900 },
      ],
      expect.any(AbortSignal),
    )
    expect(await within(court).findByTestId('court-view')).toBeVisible()
    expect(screen.queryByTestId('court-marker-surface')).not.toBeInTheDocument()
  })
})
