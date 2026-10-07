import { render, screen, within } from '@testing-library/react'
import { afterEach, vi } from 'vitest'
import App from './App'

function mockHealthOk() {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: 'ok', version: '0.1.0' }), { status: 200 }),
    ),
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('App shell', () => {
  it('renders the main landmarks', async () => {
    mockHealthOk()
    render(<App />)
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { level: 1, name: 'New analysis' })).toBeInTheDocument()
    expect(screen.getByRole('main')).toBeInTheDocument()
    await screen.findByText('API online')
  })

  it('marks Review as the current page', async () => {
    mockHealthOk()
    render(<App />)
    const nav = screen.getByRole('navigation', { name: 'Main' })
    expect(within(nav).getByRole('link', { current: 'page' })).toHaveTextContent('Review')
    await screen.findByText('API online')
  })

  it('shows the empty states for video, court and timeline', async () => {
    mockHealthOk()
    render(<App />)
    const video = screen.getByRole('region', { name: 'Rally video' })
    expect(within(video).getByText('Drop a rally video here')).toBeInTheDocument()
    expect(within(video).getByRole('button', { name: 'Play' })).toBeDisabled()

    const court = screen.getByRole('region', { name: 'Court view' })
    expect(within(court).getByText('No analysis yet')).toBeInTheDocument()

    expect(screen.getByRole('region', { name: 'Rally timeline' })).toBeInTheDocument()
    await screen.findByText('API online')
  })

  it('starts the workflow on the Upload step', async () => {
    mockHealthOk()
    render(<App />)
    const steps = screen.getByRole('list', { name: 'Analysis progress' })
    expect(within(steps).getByText('Upload').closest('li')).toHaveAttribute('aria-current', 'step')
    await screen.findByText('API online')
  })

  it('shows API offline when the health check fails', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network error')))
    render(<App />)
    expect(await screen.findByText('API offline')).toBeInTheDocument()
  })
})
