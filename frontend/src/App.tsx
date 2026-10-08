import { Upload } from 'lucide-react'
import { useRef, useState, type ChangeEvent } from 'react'
import styles from './App.module.css'
import { AppShell } from './components/layout/AppShell'
import { Header } from './components/layout/Header'
import { Sidebar } from './components/layout/Sidebar'
import { Button } from './components/ui/Button'
import { headerTextFor, reviewStepFor } from './features/analysis/presentation'
import { useAnalysis } from './features/analysis/useAnalysis'
import { CourtPanel } from './features/court/CourtPanel'
import { RallyTimeline } from './features/review/RallyTimeline'
import { ReviewSteps } from './features/review/ReviewSteps'
import { useVideoFile } from './features/video/useVideoFile'
import { VIDEO_ACCEPT_ATTR } from './features/video/videoFile'
import { useVideoPlayer } from './features/video/useVideoPlayer'
import { VideoPanel } from './features/video/VideoPanel'
import { useBackendStatus } from './hooks/useBackendStatus'

function App() {
  const backendStatus = useBackendStatus()
  const { video, error, selectFile, clear, reportPlaybackError } = useVideoFile()
  const analysis = useAnalysis(video)
  // One clock for the whole review: the video element drives the court and timeline.
  const videoRef = useRef<HTMLVideoElement>(null)
  const player = useVideoPlayer(videoRef, video?.url ?? null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  // Manual court marking applies to the video it was started for.
  const [markingFor, setMarkingFor] = useState<string | null>(null)
  const marking = video !== null && markingFor === video.url

  const openFilePicker = () => fileInputRef.current?.click()

  const onFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) selectFile(file)
    // Reset so choosing the same file again still fires a change event.
    e.target.value = ''
  }

  const { title, subtitle } = headerTextFor(analysis.state, video?.file.name ?? null)
  const result = analysis.state.phase === 'ready' ? analysis.state.result : null
  const canMarkCourt =
    (analysis.state.phase === 'failed' && Boolean(analysis.state.jobId)) ||
    (result?.source === 'cv' && analysis.state.phase === 'ready')
  const startMarking = canMarkCourt && video ? () => setMarkingFor(video.url) : undefined

  return (
    <AppShell
      sidebar={<Sidebar backendStatus={backendStatus} />}
      header={
        <Header
          title={title}
          center={<ReviewSteps current={reviewStepFor(analysis.state)} />}
          actions={
            <Button
              variant={video ? 'secondary' : 'primary'}
              icon={<Upload size={16} />}
              aria-label={video ? 'Upload another rally' : 'Upload rally'}
              onClick={openFilePicker}
            >
              <span className={styles.uploadLabel}>
                {video ? 'Upload another' : 'Upload rally'}
              </span>
            </Button>
          }
        />
      }
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={VIDEO_ACCEPT_ATTR}
        className="visually-hidden"
        tabIndex={-1}
        aria-hidden="true"
        aria-label="Choose video file"
        onChange={onFileInputChange}
      />
      <div className={styles.workspace}>
        <VideoPanel
          video={video}
          videoRef={videoRef}
          player={player}
          error={error}
          onBrowse={openFilePicker}
          onFile={selectFile}
          onClear={clear}
          onPlaybackError={reportPlaybackError}
          overlay={
            result?.calibration ? { analysis: result, calibration: result.calibration } : null
          }
          marking={
            marking
              ? {
                onSubmit: (corners) => {
                  setMarkingFor(null)
                  analysis.calibrate(corners)
                },
                onCancel: () => setMarkingFor(null),
              }
              : null
          }
        />
        <CourtPanel
          analysis={analysis.state}
          time={player.currentTime}
          onStartAnalysis={analysis.start}
          onCancelAnalysis={analysis.cancel}
          onMarkCourt={startMarking}
        />
      </div>
      <RallyTimeline
        result={result}
        duration={player.duration}
        currentTime={player.currentTime}
        onSeek={player.seek}
      />
    </AppShell>
  )
}

export default App
