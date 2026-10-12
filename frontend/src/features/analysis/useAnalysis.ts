import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ApiError,
  cancelAnalysis,
  createAnalysis,
  createYouTubeAnalysis,
  getAnalysisResult,
  getAnalysisStatus,
  newJobId,
  recalibrateAnalysis,
  type AnalysisJobStatus,
  type CourtCorners,
  type StoredVideo,
} from '../../api/client'
import { formatTime } from '../../lib/format'
import type { RallyAnalysis } from '../../types/analysis'
import type { FileSource, VideoSource, YouTubeSource } from '../video/useVideoFile'
import { readVideoDuration } from './readVideoDuration'

export const POLL_INTERVAL_MS = 500
const MAX_CONSECUTIVE_POLL_ERRORS = 3

export type AnalysisState =
  | { phase: 'idle' }
  | {
      phase: 'uploading'
      progress: number
      /** What is being sent: a file upload, or a YouTube link being checked. */
      label?: string
    }
  | { phase: 'processing'; job: AnalysisJobStatus }
  | { phase: 'ready'; job: AnalysisJobStatus; result: RallyAnalysis }
  | {
      phase: 'failed'
      message: string
      during: 'upload' | 'processing'
      /** Server job id, when the failure happened after upload (allows re-runs). */
      jobId?: string
      /** Machine-readable reason, e.g. 'court_not_found'. */
      code?: string
    }
  | { phase: 'cancelled' }

export interface AnalysisControls {
  state: AnalysisState
  /**
   * The server's stored copy of the current video, once a job has reported it.
   * Kept after a failure, a cancel or a re-run, so a YouTube video stays
   * playable (e.g. to mark the court corners) when its analysis didn't finish.
   */
  video: StoredVideo | null
  /** Starts (or restarts) analysis of the current video. */
  start: () => void
  /** Re-runs the last analysis on the same uploaded video with marked court corners. */
  calibrate: (corners: CourtCorners) => void
  cancel: () => void
}

class AnalysisFailure extends Error {
  readonly during: 'upload' | 'processing'
  readonly jobId?: string
  readonly code?: string

  constructor(
    message: string,
    during: 'upload' | 'processing',
    details: { jobId?: string; code?: string } = {},
  ) {
    super(message)
    this.during = during
    this.jobId = details.jobId
    this.code = details.code ?? undefined
  }
}

function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError'
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, ms)
    signal.addEventListener(
      'abort',
      () => {
        window.clearTimeout(timer)
        reject(new DOMException('Aborted', 'AbortError'))
      },
      { once: true },
    )
  })
}

function friendlyMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message
  if (err instanceof TypeError) {
    return 'Couldn’t reach the analysis server. Check that the backend is running.'
  }
  return 'Something went wrong. Please try again.'
}

interface JobContext {
  signal: AbortSignal
  update: (state: AnalysisState) => void
  /** Id of the job about to be created (chosen here, see newJobId). */
  jobId: string
  /** Call right before the create request goes out: from then on, giving up
   *  also cancels the job on the server, even before the request returns. */
  sending: () => void
}

type StartJob = (ctx: JobContext) => Promise<AnalysisJobStatus>

/** Upload the video and start a job on it. */
function uploadJob(video: FileSource, maxVideoSeconds: number): StartJob {
  return async ({ signal, update, jobId, sending }) => {
    const durationSec = await readVideoDuration(video.url, signal)
    // Checked here too, so a long file isn't uploaded only to be refused.
    if (durationSec !== null && durationSec > maxVideoSeconds + 1) {
      throw new AnalysisFailure(
        `This video is ${formatTime(durationSec)} long. Videos up to ` +
          `${formatTime(maxVideoSeconds)} can be analysed; trim it and try again.`,
        'upload',
        { code: 'video_too_long' },
      )
    }
    try {
      sending()
      return await createAnalysis(video.file, durationSec, {
        signal,
        jobId,
        onUploadProgress: (progress) => update({ phase: 'uploading', progress }),
      })
    } catch (err) {
      if (isAbort(err)) throw err
      throw new AnalysisFailure(friendlyMessage(err), 'upload', { code: errorCode(err) })
    }
  }
}

/** Ask the server to fetch a YouTube video and analyse it. */
function youtubeJob(video: YouTubeSource): StartJob {
  return async ({ signal, update, jobId, sending }) => {
    update({ phase: 'uploading', progress: 0, label: 'Sending the link to the analyser' })
    try {
      sending()
      return await createYouTubeAnalysis(
        { url: video.info.url, startSec: video.startSec, endSec: video.endSec, jobId },
        signal,
      )
    } catch (err) {
      if (isAbort(err)) throw err
      throw new AnalysisFailure(friendlyMessage(err), 'upload', { code: errorCode(err) })
    }
  }
}

function errorCode(err: unknown): string | undefined {
  return err instanceof ApiError ? err.code : undefined
}

/** Re-run an existing job's video with manually marked court corners. */
function recalibrationJob(originalId: string, corners: CourtCorners): StartJob {
  return async ({ signal, jobId, sending }) => {
    try {
      sending()
      return await recalibrateAnalysis(originalId, corners, signal, jobId)
    } catch (err) {
      if (isAbort(err)) throw err
      throw new AnalysisFailure(friendlyMessage(err), 'processing', { jobId: originalId })
    }
  }
}

async function runAnalysis(startJob: StartJob, ctx: JobContext): Promise<void> {
  const { signal, update } = ctx
  let job = await startJob(ctx)
  update({ phase: 'processing', job })

  let errors = 0
  while (job.status === 'queued' || job.status === 'processing') {
    await sleep(POLL_INTERVAL_MS, signal)
    try {
      job = await getAnalysisStatus(job.id, signal)
      errors = 0
    } catch (err) {
      if (isAbort(err)) throw err
      if (err instanceof ApiError && err.status === 404) {
        throw new AnalysisFailure(
          'The server lost track of this analysis (it may have restarted). Please retry.',
          'processing',
        )
      }
      errors += 1
      if (errors >= MAX_CONSECUTIVE_POLL_ERRORS) {
        throw new AnalysisFailure(friendlyMessage(err), 'processing', { jobId: job.id })
      }
      continue
    }
    update({ phase: 'processing', job })
  }

  if (job.status === 'failed') {
    throw new AnalysisFailure(job.error ?? 'Analysis failed.', 'processing', {
      jobId: job.id,
      code: job.errorCode ?? undefined,
    })
  }
  try {
    const result = await getAnalysisResult(job.id, signal)
    update({ phase: 'ready', job, result })
  } catch (err) {
    if (isAbort(err)) throw err
    throw new AnalysisFailure(friendlyMessage(err), 'processing', { jobId: job.id })
  }
}

const IDLE: AnalysisState = { phase: 'idle' }
const STARTING: AnalysisState = { phase: 'uploading', progress: 0 }

/**
 * Drives Upload → Processing → Ready for the current video. Analysis starts
 * automatically when a video is chosen (a file, or a YouTube link) and is
 * abandoned when it changes.
 *
 * Stored state is tagged with the source it belongs to, so a stale result can
 * never be shown against a different video.
 */
export function useAnalysis(
  source: VideoSource | null,
  maxVideoSeconds: number,
): AnalysisControls {
  const [tagged, setTagged] = useState<{
    key: string
    state: AnalysisState
    video: StoredVideo | null
  } | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const jobIdRef = useRef<string | null>(null)

  /** Shows `state` for the source `key`, keeping the stored video already known for it. */
  const show = useCallback((key: string, state: AnalysisState) => {
    const reported = state.phase === 'processing' || state.phase === 'ready' ? state.job.video : null
    setTagged((prev) => ({
      key,
      state,
      video: reported ?? (prev?.key === key ? prev.video : null),
    }))
  }, [])

  const run = useCallback((target: VideoSource, startJob: StartJob) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    jobIdRef.current = null
    const jobId = newJobId()
    const update = (next: AnalysisState) => {
      if (controller.signal.aborted) return
      // Remember the job only while it is being created or runs: that is what
      // a cancel should stop.
      if (next.phase === 'processing') jobIdRef.current = next.job.id
      else if (next.phase !== 'uploading') jobIdRef.current = null
      show(target.key, next)
    }
    const sending = () => {
      if (!controller.signal.aborted) jobIdRef.current = jobId
    }

    runAnalysis(startJob, { signal: controller.signal, update, jobId, sending }).catch((err: unknown) => {
      if (controller.signal.aborted || isAbort(err)) return
      if (!(err instanceof AnalysisFailure)) console.error(err)
      update(
        err instanceof AnalysisFailure
          ? {
              phase: 'failed',
              message: err.message,
              during: err.during,
              jobId: err.jobId,
              code: err.code,
            }
          : { phase: 'failed', message: friendlyMessage(err), during: 'processing' },
      )
    })
  }, [show])

  const jobFor = useCallback(
    (target: VideoSource): StartJob =>
      target.kind === 'file' ? uploadJob(target, maxVideoSeconds) : youtubeJob(target),
    [maxVideoSeconds],
  )

  useEffect(() => {
    if (!source) return
    run(source, jobFor(source))
    return () => {
      controllerRef.current?.abort()
      stopOnServer(jobIdRef.current)
    }
  }, [source, run, jobFor])

  const start = useCallback(() => {
    if (!source) return
    show(source.key, STARTING)
    run(source, jobFor(source))
  }, [source, show, run, jobFor])

  const current = source && tagged?.key === source.key ? tagged : null
  const state = !source ? IDLE : current ? current.state : STARTING
  const video = current?.video ?? null

  const lastJobId =
    state.phase === 'ready' || state.phase === 'processing'
      ? state.job.id
      : state.phase === 'failed'
        ? state.jobId
        : undefined

  const calibrate = useCallback(
    (corners: CourtCorners) => {
      if (!source || !lastJobId) return
      show(source.key, { phase: 'processing', job: pendingJob(lastJobId) })
      run(source, recalibrationJob(lastJobId, corners))
    },
    [source, lastJobId, show, run],
  )

  const cancel = useCallback(() => {
    controllerRef.current?.abort()
    stopOnServer(jobIdRef.current)
    jobIdRef.current = null
    if (source) show(source.key, { phase: 'cancelled' })
  }, [source, show])

  return { state, video, start, calibrate, cancel }
}

/** Frees the server from a job nobody is waiting for any more (best effort). */
function stopOnServer(jobId: string | null) {
  if (jobId) void cancelAnalysis(jobId).catch(() => {})
}

/** Share of a YouTube job's progress given to the download (backend: jobs.DOWNLOAD_SHARE). */
const YOUTUBE_DOWNLOAD_SHARE = 0.15

/** For a YouTube video still downloading: how far the download is (0-1), else null. */
export function downloadProgressFor(
  source: VideoSource | null,
  state: AnalysisState,
): number | null {
  if (source?.kind !== 'youtube' || state.phase !== 'processing') return null
  const { job } = state
  if (job.video || job.currentStage !== 0) return null
  return Math.min(job.progress / YOUTUBE_DOWNLOAD_SHARE, 1)
}

/**
 * Where to play the video: the local file, or the server's copy of a YouTube
 * video (`video`, from useAnalysis) once it has been downloaded.
 */
export function playbackUrlFor(
  source: VideoSource | null,
  video: StoredVideo | null,
): string | null {
  if (!source) return null
  if (source.kind === 'file') return source.url
  return video?.url ?? null
}

/** Whether the analysis is still under way (a YouTube video may still be downloading). */
export function isAnalysisActive(state: AnalysisState): boolean {
  return state.phase === 'uploading' || state.phase === 'processing'
}

/** Placeholder status shown between submitting corners and the server's reply. */
function pendingJob(id: string): AnalysisJobStatus {
  return { id, status: 'queued', progress: 0, stages: [], currentStage: null, error: null }
}
