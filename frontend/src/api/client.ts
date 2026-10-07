/**
 * Thin HTTP client for the RallyReview backend. Components should go through
 * this module rather than calling fetch directly, so the transport can change
 * (or be mocked) without touching UI code.
 */

import type { ImagePoint, RallyAnalysis } from '../types/analysis'

const API_BASE = '/api'

export interface HealthResponse {
  status: 'ok'
  version: string
}

export type AnalysisJobState = 'queued' | 'processing' | 'ready' | 'failed'

export interface AnalysisJobStatus {
  id: string
  status: AnalysisJobState
  /** Overall progress, 0 to 1. */
  progress: number
  /** Pipeline stage labels, in order. */
  stages: string[]
  /** Index into `stages` while processing, otherwise null. */
  currentStage: number | null
  error: string | null
  /** Machine-readable failure reason, e.g. 'court_not_found'. */
  errorCode?: string | null
}

/** The four outer court corners in normalized image coordinates, ordered
 *  near-left, near-right, far-right, far-left (near = closest to the camera). */
export type CourtCorners = [ImagePoint, ImagePoint, ImagePoint, ImagePoint]

/** An HTTP error response from the API, with the server's `detail` message if any. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function detailFrom(body: unknown): string | null {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
  }
  return null
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { signal })
  const body: unknown = await res.json().catch(() => null)
  if (!res.ok) {
    throw new ApiError(res.status, detailFrom(body) ?? `Request failed (HTTP ${res.status})`)
  }
  return body as T
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return getJson('/health', signal)
}

export function getAnalysisStatus(id: string, signal?: AbortSignal): Promise<AnalysisJobStatus> {
  return getJson(`/analyses/${encodeURIComponent(id)}`, signal)
}

export function getAnalysisResult(id: string, signal?: AbortSignal): Promise<RallyAnalysis> {
  return getJson(`/analyses/${encodeURIComponent(id)}/result`, signal)
}

/** URL of the raw analysis JSON, for debugging and CV integration work. */
export function analysisResultUrl(id: string): string {
  return `${API_BASE}/analyses/${encodeURIComponent(id)}/result`
}

interface CreateAnalysisOptions {
  signal?: AbortSignal
  /** Upload progress, 0 to 1. */
  onUploadProgress?: (fraction: number) => void
}

/**
 * Uploads a video and starts an analysis job. Uses XMLHttpRequest because
 * fetch cannot report upload progress.
 */
export function createAnalysis(
  file: File,
  durationSec: number | null,
  { signal, onUploadProgress }: CreateAnalysisOptions = {},
): Promise<AnalysisJobStatus> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'))
      return
    }

    const form = new FormData()
    form.append('video', file)
    if (durationSec !== null) form.append('duration_sec', String(durationSec))

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/analyses`)
    xhr.responseType = 'json'

    const onAbort = () => xhr.abort()
    signal?.addEventListener('abort', onAbort, { once: true })
    const done = () => signal?.removeEventListener('abort', onAbort)

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onUploadProgress?.(e.loaded / e.total)
    }
    xhr.onload = () => {
      done()
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as AnalysisJobStatus)
      } else {
        reject(
          new ApiError(
            xhr.status,
            detailFrom(xhr.response) ?? `Upload failed (HTTP ${xhr.status})`,
          ),
        )
      }
    }
    xhr.onerror = () => {
      done()
      reject(new TypeError('Network error during upload'))
    }
    xhr.onabort = () => {
      done()
      reject(new DOMException('Aborted', 'AbortError'))
    }

    xhr.send(form)
  })
}

/** Re-runs an analysis on the same uploaded video with manually marked court corners. */
export async function recalibrateAnalysis(
  id: string,
  corners: CourtCorners,
  signal?: AbortSignal,
): Promise<AnalysisJobStatus> {
  const res = await fetch(`${API_BASE}/analyses/${encodeURIComponent(id)}/calibration`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ corners }),
    signal,
  })
  const body: unknown = await res.json().catch(() => null)
  if (!res.ok) {
    throw new ApiError(res.status, detailFrom(body) ?? `Request failed (HTTP ${res.status})`)
  }
  return body as AnalysisJobStatus
}
