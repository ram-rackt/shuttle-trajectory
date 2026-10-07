import type { AnalysisJobStatus } from '../../api/client'
import { headerTextFor, reviewStepFor } from './presentation'

const job: AnalysisJobStatus = {
  id: 'j',
  status: 'processing',
  progress: 0.3,
  stages: ['Detecting court', 'Tracking shuttle'],
  currentStage: 1,
  error: null,
}

describe('reviewStepFor', () => {
  it.each([
    [{ phase: 'idle' } as const, 'upload'],
    [{ phase: 'uploading', progress: 0.2 } as const, 'upload'],
    [{ phase: 'processing', job } as const, 'process'],
    [{ phase: 'failed', message: 'x', during: 'upload' } as const, 'upload'],
    [{ phase: 'failed', message: 'x', during: 'processing' } as const, 'process'],
    [{ phase: 'cancelled' } as const, 'upload'],
  ])('maps %o to %s', (state, step) => {
    expect(reviewStepFor(state)).toBe(step)
  })
})

describe('headerTextFor', () => {
  it('prompts for upload when there is no video', () => {
    expect(headerTextFor({ phase: 'idle' }, null).title).toBe('New analysis')
  })

  it('names the current stage while processing', () => {
    expect(headerTextFor({ phase: 'processing', job }, 'r.mp4')).toEqual({
      title: 'Analysing rally',
      subtitle: 'r.mp4 · Tracking shuttle',
    })
  })

  it('shows "Queued" before the first stage starts', () => {
    const queued = { ...job, status: 'queued' as const, currentStage: null }
    expect(headerTextFor({ phase: 'processing', job: queued }, 'r.mp4').subtitle).toBe(
      'r.mp4 · Queued',
    )
  })
})
