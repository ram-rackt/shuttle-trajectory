import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ApiError,
  createAnalysis,
  getAnalysisResult,
  getAnalysisStatus,
  recalibrateAnalysis,
  type AnalysisJobStatus,
  type CourtCorners,
} from '../../api/client'
import type { RallyAnalysis } from '../../types/analysis'
import type { LoadedVideo } from '../video/useVideoFile'
import { readVideoDuration } from './readVideoDuration'

export const POLL_INTERVAL_MS = 500
const MAX_CONSECUTIVE_POLL_ERRORS = 3

export type AnalysisState =
  | { phase: 'idle' }
  | { phase: 'uploading'; progress: number }
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

type StartJob = (
  signal: AbortSignal,
  update: (state: AnalysisState) => void,
) => Promise<AnalysisJobStatus>

/** Upload the video and start a job on it. */
function uploadJob(video: LoadedVideo): StartJob {
  return async (signal, update) => {
    const durationSec = await readVideoDuration(video.url, signal)
    try {
      return await createAnalysis(video.file, durationSec, {
        signal,
        onUploadProgress: (progress) => update({ phase: 'uploading', progress }),
      })
    } catch (err) {
      if (isAbort(err)) throw err
      throw new AnalysisFailure(friendlyMessage(err), 'upload')
    }
  }
}

/** Re-run an existing job's video with manually marked court corners. */
function recalibrationJob(jobId: string, corners: CourtCorners): StartJob {
  return async (signal) => {
    try {
      return await recalibrateAnalysis(jobId, corners, signal)
    } catch (err) {
      if (isAbort(err)) throw err
      throw new AnalysisFailure(friendlyMessage(err), 'processing', { jobId })
    }
  }
}

async function runAnalysis(
  startJob: StartJob,
  signal: AbortSignal,
  update: (state: AnalysisState) => void,
): Promise<void> {
  let job = await startJob(signal, update)
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
 * automatically when a video is loaded and is abandoned when it changes.
 *
 * Stored state is tagged with the video URL it belongs to, so a stale result
 * can never be shown against a different video.
 */
export function useAnalysis(video: LoadedVideo | null): AnalysisControls {
  const [tagged, setTagged] = useState<{ url: string; state: AnalysisState } | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  const run = useCallback((target: LoadedVideo, startJob: StartJob) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    const update = (next: AnalysisState) => {
      if (!controller.signal.aborted) setTagged({ url: target.url, state: next })
    }

    runAnalysis(startJob, controller.signal, update).catch((err: unknown) => {
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
  }, [])

  useEffect(() => {
    if (!video) return
    run(video, uploadJob(video))
    return () => controllerRef.current?.abort()
  }, [video, run])

  const start = useCallback(() => {
    if (!video) return
    setTagged({ url: video.url, state: STARTING })
    run(video, uploadJob(video))
  }, [video, run])

  const state = !video ? IDLE : tagged?.url === video.url ? tagged.state : STARTING

  const lastJobId =
    state.phase === 'ready' || state.phase === 'processing'
      ? state.job.id
      : state.phase === 'failed'
        ? state.jobId
        : undefined

  const calibrate = useCallback(
    (corners: CourtCorners) => {
      if (!video || !lastJobId) return
      setTagged({
        url: video.url,
        state: { phase: 'processing', job: pendingJob(lastJobId) },
      })
      run(video, recalibrationJob(lastJobId, corners))
    },
    [video, lastJobId, run],
  )

  const cancel = useCallback(() => {
    controllerRef.current?.abort()
    if (video) setTagged({ url: video.url, state: { phase: 'cancelled' } })
  }, [video])

  return { state, start, calibrate, cancel }
}

/** Placeholder status shown between submitting corners and the server's reply. */
function pendingJob(id: string): AnalysisJobStatus {
  return { id, status: 'queued', progress: 0, stages: [], currentStage: null, error: null }
}
