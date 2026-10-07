import type { ReviewStep } from '../review/ReviewSteps'
import type { AnalysisState } from './useAnalysis'

/** Which workflow step the header stepper should highlight. */
export function reviewStepFor(state: AnalysisState): ReviewStep {
  switch (state.phase) {
    case 'processing':
      return 'process'
    case 'failed':
      return state.during === 'processing' ? 'process' : 'upload'
    case 'ready':
      return 'review'
    default:
      return 'upload'
  }
}

/** Header title and subtitle for the current state. */
export function headerTextFor(
  state: AnalysisState,
  fileName: string | null,
): { title: string; subtitle: string } {
  if (!fileName) {
    return { title: 'New analysis', subtitle: 'Upload a rally video to map it onto the court' }
  }
  switch (state.phase) {
    case 'uploading':
      return { title: 'Uploading rally', subtitle: fileName }
    case 'processing': {
      const { job } = state
      const stage = job.currentStage !== null ? job.stages[job.currentStage] : 'Queued'
      return { title: 'Analysing rally', subtitle: `${fileName} · ${stage}` }
    }
    case 'ready': {
      const n = state.result.shots.length
      return { title: 'Rally review', subtitle: `${fileName} · ${n} ${n === 1 ? 'shot' : 'shots'}` }
    }
    case 'failed':
      return { title: 'Analysis failed', subtitle: fileName }
    default:
      return { title: 'Rally loaded', subtitle: fileName }
  }
}
