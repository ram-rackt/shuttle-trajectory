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

/** A video stored on the server (uploaded, or downloaded from YouTube). */
export interface StoredVideo {
  id: string
  /** Where the browser can play it (supports seeking). */
  url: string
  name: string
  sizeBytes: number
  source: 'upload' | 'youtube'
  youtubeUrl?: string | null
  channel?: string | null
}

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
  /** The result was reused from an earlier identical analysis. */
  cached?: boolean
  /** The stored video; for YouTube jobs, once it has been downloaded. */
  video?: StoredVideo | null
  /** The part of the video analysed, when not all of it. */
  window?: { startSec: number; endSec: number } | null
}

/** Limits the server enforces; the UI checks them before sending anything. */
export interface ServerConfig {
  maxUploadBytes: number
  maxVideoSeconds: number
  youtube: { enabled: boolean; maxSourceSeconds: number }
}

export const DEFAULT_SERVER_CONFIG: ServerConfig = {
  maxUploadBytes: 2 * 1024 ** 3,
  maxVideoSeconds: 11 * 60,
  youtube: { enabled: true, maxSourceSeconds: 3 * 60 * 60 },
}

/** What the server found out about a YouTube link before analysing it. */
export interface YouTubeInfo {
  id: string
  url: string
  title: string
  channel: string | null
  durationSec: number
  thumbnailUrl: string | null
  height: number | null
  /** Longest part analysed at once. */
  maxVideoSeconds: number
  /** Longer than maxVideoSeconds: a start and end time must be chosen. */
  needsRange: boolean
}

/**
 * A new job id (32 hex digits). The client picks the id of each job it starts,
 * so it can cancel the job even if it gives up before the server has replied
 * (e.g. while a YouTube link is still being checked).
 */
export function newJobId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

/** The four outer court corners in normalized image coordinates, ordered
 *  near-left, near-right, far-right, far-left (near = closest to the camera). */
export type CourtCorners = [ImagePoint, ImagePoint, ImagePoint, ImagePoint]

/** An HTTP error response from the API, with the server's `detail` message if any. */
export class ApiError extends Error {
  readonly status: number
  /** Machine-readable reason when the server gives one, e.g. 'video_private'. */
  readonly code?: string

  constructor(status: number, message: string, code?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

function codeFrom(body: unknown): string | undefined {
  if (body && typeof body === 'object' && 'code' in body) {
    const code = (body as { code: unknown }).code
    if (typeof code === 'string') return code
  }
  return undefined
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
  return parse<T>(res)
}

async function postJson<T>(path: string, payload: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: payload === undefined ? undefined : JSON.stringify(payload),
    signal,
  })
  return parse<T>(res)
}

async function parse<T>(res: Response): Promise<T> {
  const body: unknown = await res.json().catch(() => null)
  if (!res.ok) {
    throw new ApiError(
      res.status,
      detailFrom(body) ?? `Request failed (HTTP ${res.status})`,
      codeFrom(body),
    )
  }
  return body as T
}

export function getServerConfig(signal?: AbortSignal): Promise<ServerConfig> {
  return getJson('/config', signal)
}

/** Checks a YouTube link: title, length and whether it can be analysed. */
export function getYouTubeInfo(url: string, signal?: AbortSignal): Promise<YouTubeInfo> {
  return getJson(`/youtube/info?url=${encodeURIComponent(url)}`, signal)
}

export interface YouTubeAnalysisOptions {
  url: string
  /** Analyse only this part (required for videos longer than maxVideoSeconds). */
  startSec?: number | null
  endSec?: number | null
  /** Id for the new job (see newJobId). */
  jobId?: string
}

/** Starts an analysis of a YouTube video; the server downloads it first. */
export function createYouTubeAnalysis(
  { url, startSec = null, endSec = null, jobId }: YouTubeAnalysisOptions,
  signal?: AbortSignal,
): Promise<AnalysisJobStatus> {
  return postJson('/analyses/youtube', { url, startSec, endSec, jobId }, signal)
}

/** Asks the server to stop a job (best effort: the UI stops waiting regardless). */
export function cancelAnalysis(id: string): Promise<AnalysisJobStatus> {
  return postJson(`/analyses/${encodeURIComponent(id)}/cancel`, undefined)
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
  /** Id for the new job (see newJobId). */
  jobId?: string
}

/**
 * Uploads a video and starts an analysis job. Uses XMLHttpRequest because
 * fetch cannot report upload progress.
 */
export function createAnalysis(
  file: File,
  durationSec: number | null,
  { signal, onUploadProgress, jobId }: CreateAnalysisOptions = {},
): Promise<AnalysisJobStatus> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'))
      return
    }

    // The small fields go first: the server checks them (e.g. the length
    // limit) before it stores any of the video.
    const form = new FormData()
    if (jobId) form.append('job_id', jobId)
    if (durationSec !== null) form.append('duration_sec', String(durationSec))
    form.append('video', file)

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
            codeFrom(xhr.response),
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
export function recalibrateAnalysis(
  id: string,
  corners: CourtCorners,
  signal?: AbortSignal,
  jobId?: string,
): Promise<AnalysisJobStatus> {
  return postJson(`/analyses/${encodeURIComponent(id)}/calibration`, { corners, jobId }, signal)
}
