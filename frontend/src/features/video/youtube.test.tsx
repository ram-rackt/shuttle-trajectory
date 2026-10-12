import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import App from '../../App'
import * as api from '../../api/client'
import { normalizeConfig } from '../../hooks/useServerConfig'
import type { RallyAnalysis } from '../../types/analysis'
import { rangeProblem } from './youtubeRange'

vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof api>()
  return {
    ...actual,
    getHealth: vi.fn().mockResolvedValue({ status: 'ok', version: 'test' }),
    getServerConfig: vi.fn(),
    getYouTubeInfo: vi.fn(),
    createYouTubeAnalysis: vi.fn(),
    getAnalysisStatus: vi.fn(),
    getAnalysisResult: vi.fn(),
    cancelAnalysis: vi.fn().mockResolvedValue({}),
  }
})

const INFO: api.YouTubeInfo = {
  id: 'E3V9GWOr-_o',
  url: 'https://www.youtube.com/watch?v=E3V9GWOr-_o',
  title: 'Play of the Day',
  channel: 'BWF TV',
  durationSec: 128,
  thumbnailUrl: 'https://i.ytimg.com/vi/E3V9GWOr-_o/hqdefault.jpg',
  height: 1080,
  maxVideoSeconds: 660,
  needsRange: false,
}

const STAGES = ['Downloading video', 'Detecting court', 'Tracking', 'Shots', 'Mapping']

function job(over: Partial<api.AnalysisJobStatus>): api.AnalysisJobStatus {
  return {
    id: 'job-yt',
    status: 'processing',
    progress: 0,
    stages: STAGES,
    currentStage: 0,
    error: null,
    ...over,
  }
}

const STORED: api.StoredVideo = {
  id: 'a'.repeat(64),
  url: `/api/videos/${'a'.repeat(64)}/file`,
  name: 'Play of the Day',
  sizeBytes: 24_000_000,
  source: 'youtube',
  youtubeUrl: INFO.url,
}

const RESULT: RallyAnalysis = {
  schemaVersion: 1,
  durationSec: 128,
  players: [],
  shuttle: [],
  shots: [{ index: 0, hitter: 'A', type: 'serve', startTime: 1, endTime: 2 }],
}

function setConfig(youtubeEnabled = true) {
  vi.mocked(api.getServerConfig).mockResolvedValue({
    maxUploadBytes: 2 * 1024 ** 3,
    maxVideoSeconds: 660,
    youtube: { enabled: youtubeEnabled, maxSourceSeconds: 10800 },
  })
}

async function openYouTubeTab() {
  const user = userEvent.setup()
  render(<App />)
  await screen.findByText('API online')
  await user.click(await screen.findByRole('tab', { name: 'YouTube link' }))
  return user
}

async function checkLink(user: ReturnType<typeof userEvent.setup>, url = INFO.url) {
  await user.type(screen.getByLabelText('Link to a YouTube video'), url)
  await user.click(screen.getByRole('button', { name: 'Check link' }))
}

beforeEach(() => {
  vi.clearAllMocks()
  setConfig()
})

describe('YouTube links', () => {
  it('checks a link, downloads, then plays the server copy and shows the analysis', async () => {
    vi.mocked(api.getYouTubeInfo).mockResolvedValue(INFO)
    vi.mocked(api.createYouTubeAnalysis).mockResolvedValue(job({ progress: 0.06 }))
    vi.mocked(api.getAnalysisStatus)
      .mockResolvedValueOnce(job({ progress: 0.06 }))
      .mockResolvedValueOnce(job({ progress: 0.5, currentStage: 2, video: STORED }))
      .mockResolvedValue(job({ status: 'ready', progress: 1, currentStage: null, video: STORED }))
    vi.mocked(api.getAnalysisResult).mockResolvedValue(RESULT)

    const user = await openYouTubeTab()
    await checkLink(user)
    const card = await screen.findByRole('form', { name: 'YouTube video' })
    expect(within(card).getByText('Play of the Day')).toBeInTheDocument()
    expect(within(card).getByText('BWF TV · 2:08 · 1080p')).toBeInTheDocument()
    expect(within(card).getByRole('link', { name: /Open on YouTube/ })).toHaveAttribute(
      'href',
      INFO.url,
    )

    await user.click(within(card).getByRole('button', { name: 'Analyse video' }))
    expect(api.createYouTubeAnalysis).toHaveBeenCalledWith(
      {
        url: INFO.url,
        startSec: null,
        endSec: null,
        jobId: expect.stringMatching(/^[0-9a-f]{32}$/),
      },
      expect.anything(),
    )
    // Downloading: thumbnail and progress (6% of the job = 40% of the download).
    const pending = await screen.findByTestId('youtube-pending')
    expect(pending).toHaveTextContent('Fetching the video from YouTube')
    expect(pending).toHaveTextContent('40% downloaded')

    // Downloaded: the player plays the server's copy, even while analysis continues.
    const video = await screen.findByTestId<HTMLVideoElement>('rally-video', {}, { timeout: 3000 })
    expect(video.getAttribute('src')).toBe(STORED.url)
    const header = screen.getByRole('region', { name: 'Rally video' })
    expect(within(header).getByRole('link', { name: /Play of the Day/ })).toHaveAttribute(
      'href',
      INFO.url,
    )
    expect(await screen.findByTestId('court-view', {}, { timeout: 3000 })).toBeInTheDocument()
  })

  it('requires a part of long videos and validates it', async () => {
    vi.mocked(api.getYouTubeInfo).mockResolvedValue({
      ...INFO,
      durationSec: 40 * 60,
      needsRange: true,
    })
    vi.mocked(api.createYouTubeAnalysis).mockResolvedValue(job({}))
    vi.mocked(api.getAnalysisStatus).mockResolvedValue(job({}))

    const user = await openYouTubeTab()
    await checkLink(user)
    const card = await screen.findByRole('form', { name: 'YouTube video' })
    expect(within(card).getByRole('radio', { name: 'Whole video' })).toBeDisabled()
    expect(within(card).getByRole('radio', { name: 'Part of it' })).toBeChecked()
    expect(within(card).getByText(/longer than 11:00/)).toBeInTheDocument()

    const end = within(card).getByLabelText('to')
    expect(end).toHaveValue('11:00')
    await user.clear(end)
    await user.type(end, '12:30')
    expect(within(card).getByRole('alert')).toHaveTextContent('Choose at most 11:00')
    expect(within(card).getByRole('button', { name: 'Analyse video' })).toBeDisabled()

    const start = within(card).getByLabelText('From')
    await user.clear(start)
    await user.type(start, '2:00')
    await user.click(within(card).getByRole('button', { name: 'Analyse video' }))
    expect(api.createYouTubeAnalysis).toHaveBeenCalledWith(
      { url: INFO.url, startSec: 120, endSec: 750, jobId: expect.any(String) },
      expect.anything(),
    )
  })

  it('shows why a link can’t be used', async () => {
    vi.mocked(api.getYouTubeInfo).mockRejectedValue(
      new api.ApiError(422, 'This video is private.', 'video_private'),
    )
    const user = await openYouTubeTab()
    await checkLink(user)
    expect(await screen.findByRole('alert')).toHaveTextContent('This video is private.')
    expect(screen.getByLabelText('Link to a YouTube video')).toHaveAttribute('aria-invalid', 'true')
    expect(api.createYouTubeAnalysis).not.toHaveBeenCalled()
  })

  it('reports a failed download in the court panel', async () => {
    vi.mocked(api.getYouTubeInfo).mockResolvedValue(INFO)
    vi.mocked(api.createYouTubeAnalysis).mockResolvedValue(job({}))
    vi.mocked(api.getAnalysisStatus).mockResolvedValue(
      job({
        status: 'failed',
        currentStage: null,
        error: 'YouTube refused the download from this server.',
        errorCode: 'youtube_blocked',
      }),
    )
    const user = await openYouTubeTab()
    await checkLink(user)
    await user.click(await screen.findByRole('button', { name: 'Analyse video' }))
    const court = screen.getByRole('region', { name: 'Court view' })
    expect(
      await within(court).findByText(/YouTube refused the download/, {}, { timeout: 3000 }),
    ).toBeInTheDocument()
    // Nothing is being fetched any more: no endless "Fetching…" spinner.
    const pending = screen.getByTestId('youtube-pending')
    expect(pending).toHaveTextContent(/video wasn.t downloaded/)
    expect(pending).not.toHaveTextContent('Fetching the video from YouTube')
  })

  it('keeps the downloaded video playable after a failure, so the court can be marked', async () => {
    vi.mocked(api.getYouTubeInfo).mockResolvedValue(INFO)
    vi.mocked(api.createYouTubeAnalysis).mockResolvedValue(job({ progress: 0.06 }))
    vi.mocked(api.getAnalysisStatus)
      .mockResolvedValueOnce(job({ progress: 0.3, currentStage: 1, video: STORED }))
      // The failure report itself doesn't repeat the video: the UI must remember it.
      .mockResolvedValue(
        job({
          status: 'failed',
          currentStage: null,
          error: 'The court lines couldn’t be found automatically.',
          errorCode: 'court_not_found',
        }),
      )

    const user = await openYouTubeTab()
    await checkLink(user)
    await user.click(await screen.findByRole('button', { name: 'Analyse video' }))
    const court = screen.getByRole('region', { name: 'Court view' })
    expect(
      await within(court).findByRole('button', { name: 'Mark court corners' }, { timeout: 3000 }),
    ).toBeInTheDocument()
    expect(screen.getByTestId<HTMLVideoElement>('rally-video').getAttribute('src')).toBe(STORED.url)
    expect(screen.queryByTestId('youtube-pending')).not.toBeInTheDocument()
  })

  it('cancelling while the link is being sent also stops the job on the server', async () => {
    vi.mocked(api.getYouTubeInfo).mockResolvedValue(INFO)
    vi.mocked(api.createYouTubeAnalysis).mockReturnValue(new Promise(() => {})) // still checking
    const user = await openYouTubeTab()
    await checkLink(user)
    await user.click(await screen.findByRole('button', { name: 'Analyse video' }))

    const court = screen.getByRole('region', { name: 'Court view' })
    await user.click(await within(court).findByRole('button', { name: 'Cancel' }))
    const sentId = vi.mocked(api.createYouTubeAnalysis).mock.calls[0][0].jobId
    expect(sentId).toMatch(/^[0-9a-f]{32}$/)
    expect(api.cancelAnalysis).toHaveBeenCalledWith(sentId)
  })

  it('hides the YouTube option when the server has it turned off', async () => {
    setConfig(false)
    render(<App />)
    await screen.findByText('API online')
    await waitFor(() => expect(api.getServerConfig).toHaveBeenCalled())
    await waitFor(() =>
      expect(screen.queryByRole('tab', { name: 'YouTube link' })).not.toBeInTheDocument(),
    )
    expect(screen.getByText('Drop a rally video here')).toBeInTheDocument()
  })
})

describe('rangeProblem', () => {
  it.each([
    [0, 60, null],
    [null, 60, 'Enter times like 1:30 or 1:02:30.'],
    [60, 30, 'The end must be after the start.'],
    [10, 11, 'Choose at least 2 seconds.'],
    [0, 700, 'Choose at most 11:00.'],
    [100, 1300, 'The video is only 20:00 long.'],
  ])('start %s, end %s', (start, end, expected) => {
    expect(rangeProblem(start, end, 1200, 660)).toBe(expected)
  })
})

describe('normalizeConfig', () => {
  it('keeps defaults for anything missing or malformed', () => {
    expect(normalizeConfig({ status: 'ok' })).toEqual(api.DEFAULT_SERVER_CONFIG)
    expect(normalizeConfig(null)).toEqual(api.DEFAULT_SERVER_CONFIG)
    const cfg = normalizeConfig({ maxVideoSeconds: 300, youtube: { enabled: false } })
    expect(cfg.maxVideoSeconds).toBe(300)
    expect(cfg.youtube.enabled).toBe(false)
    expect(cfg.maxUploadBytes).toBe(api.DEFAULT_SERVER_CONFIG.maxUploadBytes)
    expect(normalizeConfig({ maxUploadBytes: -5 }).maxUploadBytes).toBe(
      api.DEFAULT_SERVER_CONFIG.maxUploadBytes,
    )
  })
})
