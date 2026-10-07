const TIMEOUT_MS = 5000

/**
 * Reads a video's duration from its metadata without rendering it.
 * Resolves null if the browser can't tell (unsupported codec, timeout).
 */
export function readVideoDuration(url: string, signal?: AbortSignal): Promise<number | null> {
  return new Promise((resolve) => {
    const video = document.createElement('video')
    let timer = 0

    const finish = (value: number | null) => {
      window.clearTimeout(timer)
      signal?.removeEventListener('abort', onAbort)
      video.removeAttribute('src')
      video.load()
      resolve(value)
    }
    const onAbort = () => finish(null)

    video.preload = 'metadata'
    video.muted = true
    video.onloadedmetadata = () =>
      finish(Number.isFinite(video.duration) && video.duration > 0 ? video.duration : null)
    video.onerror = () => finish(null)
    signal?.addEventListener('abort', onAbort, { once: true })
    timer = window.setTimeout(() => finish(null), TIMEOUT_MS)
    video.src = url
  })
}
