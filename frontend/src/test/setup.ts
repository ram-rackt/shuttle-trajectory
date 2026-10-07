import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

// jsdom doesn't implement media playback or object URLs.
URL.createObjectURL = vi.fn(() => 'blob:mock-video')
URL.revokeObjectURL = vi.fn()

Object.defineProperty(HTMLMediaElement.prototype, 'play', {
  configurable: true,
  value: vi.fn(function (this: HTMLMediaElement) {
    Object.defineProperty(this, 'paused', { configurable: true, value: false })
    this.dispatchEvent(new Event('play'))
    return Promise.resolve()
  }),
})

Object.defineProperty(HTMLMediaElement.prototype, 'pause', {
  configurable: true,
  value: vi.fn(function (this: HTMLMediaElement) {
    Object.defineProperty(this, 'paused', { configurable: true, value: true })
    this.dispatchEvent(new Event('pause'))
  }),
})
