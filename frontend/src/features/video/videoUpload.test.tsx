import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, vi } from 'vitest'
import App from '../../App'
import * as api from '../../api/client'

// Analysis starts automatically on upload; keep it pending so these tests stay
// focused on the video panel (the flow itself is covered in analysisFlow.test).
vi.mock('../analysis/readVideoDuration', () => ({ readVideoDuration: vi.fn().mockResolvedValue(10) }))
vi.mock('../../api/client', async (importOriginal) => ({
  ...(await importOriginal<typeof api>()),
  createAnalysis: vi.fn(() => new Promise(() => {})),
}))

function videoFile(name = 'rally.mp4', type = 'video/mp4') {
  return new File(['fake video bytes'], name, { type })
}

beforeEach(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: 'ok', version: '0.1.0' }), { status: 200 }),
    ),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

async function renderApp() {
  render(<App />)
  await screen.findByText('API online')
  return screen.getByRole('region', { name: 'Rally video' })
}

describe('video upload', () => {
  it('opens the file picker from the header and the dropzone', async () => {
    const user = userEvent.setup()
    const panel = await renderApp()
    const input = screen.getByLabelText<HTMLInputElement>('Choose video file')
    const click = vi.spyOn(input, 'click')

    await user.click(screen.getByRole('button', { name: 'Upload rally' }))
    await user.click(within(panel).getByRole('button', { name: 'Choose video' }))

    expect(click).toHaveBeenCalledTimes(2)
  })

  it('previews a chosen video and shows its name', async () => {
    const user = userEvent.setup()
    const panel = await renderApp()

    await user.upload(screen.getByLabelText('Choose video file'), videoFile('final-rally.mp4'))

    const video = within(panel).getByTestId('rally-video')
    expect(video).toHaveAttribute('src', 'blob:mock-video')
    expect(within(panel).getByText('final-rally.mp4')).toBeInTheDocument()
    expect(within(panel).queryByText('Drop a rally video here')).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { level: 1, name: 'Uploading rally' })).toBeInTheDocument()
  })

  it('rejects unsupported files with an error message', async () => {
    const user = userEvent.setup({ applyAccept: false })
    const panel = await renderApp()

    await user.upload(
      screen.getByLabelText('Choose video file'),
      new File(['%PDF'], 'notes.pdf', { type: 'application/pdf' }),
    )

    expect(within(panel).getByRole('alert')).toHaveTextContent('isn’t a supported video')
    expect(within(panel).queryByTestId('rally-video')).not.toBeInTheDocument()
  })

  it('accepts a dropped video file', async () => {
    const panel = await renderApp()
    const stage = within(panel).getByTestId('video-stage')
    const dataTransfer = { files: [videoFile('dropped.mov', 'video/quicktime')], types: ['Files'] }

    fireEvent.dragEnter(stage, { dataTransfer })
    expect(within(panel).getByText('Drop to load video')).toBeInTheDocument()
    fireEvent.drop(stage, { dataTransfer })

    expect(within(panel).getByTestId('rally-video')).toBeInTheDocument()
    expect(within(panel).getByText('dropped.mov')).toBeInTheDocument()
    expect(within(panel).queryByText('Drop to load video')).not.toBeInTheDocument()
  })

  it('toggles playback from the transport controls', async () => {
    const user = userEvent.setup()
    const panel = await renderApp()
    await user.upload(screen.getByLabelText('Choose video file'), videoFile())

    await user.click(within(panel).getByRole('button', { name: 'Play' }))
    expect(within(panel).getByRole('button', { name: 'Pause' })).toBeInTheDocument()

    await user.click(within(panel).getByRole('button', { name: 'Pause' }))
    expect(within(panel).getByRole('button', { name: 'Play' })).toBeInTheDocument()
  })

  it('removes the video and returns to the empty state', async () => {
    const user = userEvent.setup()
    const panel = await renderApp()
    await user.upload(screen.getByLabelText('Choose video file'), videoFile())

    await user.click(within(panel).getByRole('button', { name: 'Remove video' }))

    expect(within(panel).getByText('Drop a rally video here')).toBeInTheDocument()
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-video')
  })

  it('shows an error when the browser cannot decode the video', async () => {
    const user = userEvent.setup()
    const panel = await renderApp()
    await user.upload(screen.getByLabelText('Choose video file'), videoFile('hevc.mov'))

    fireEvent.error(within(panel).getByTestId('rally-video'))

    expect(within(panel).getByRole('alert')).toHaveTextContent('can’t be played in this browser')
    expect(within(panel).queryByTestId('rally-video')).not.toBeInTheDocument()
  })
})
