import { Upload } from 'lucide-react'
import { useRef, useState, type ChangeEvent } from 'react'
import styles from './App.module.css'
import { AppShell } from './components/layout/AppShell'
import { Header } from './components/layout/Header'
import { Sidebar } from './components/layout/Sidebar'
import { Button } from './components/ui/Button'
import { headerTextFor, reviewStepFor } from './features/analysis/presentation'
import {
  downloadProgressFor,
  isAnalysisActive,
  playbackUrlFor,
  useAnalysis,
} from './features/analysis/useAnalysis'
import { CourtPanel } from './features/court/CourtPanel'
import { RallyTimeline } from './features/review/RallyTimeline'
import { ReviewSteps } from './features/review/ReviewSteps'
import { useVideoSource } from './features/video/useVideoFile'
import { VIDEO_ACCEPT_ATTR } from './features/video/videoFile'
import { useVideoPlayer } from './features/video/useVideoPlayer'
import { VideoPanel } from './features/video/VideoPanel'
import { useBackendStatus } from './hooks/useBackendStatus'
import { useServerConfig } from './hooks/useServerConfig'

function App() {
  const backendStatus = useBackendStatus()
  const config = useServerConfig()
  const { source, error, selectFile, selectYouTube, clear, reportPlaybackError } =
    useVideoSource(config.maxUploadBytes)
  const analysis = useAnalysis(source, config.maxVideoSeconds)
  // The local file, or the server's copy of a YouTube video once downloaded.
  const playbackUrl = playbackUrlFor(source, analysis.video)
  // One clock for the whole review: the video element drives the court and timeline.
  const videoRef = useRef<HTMLVideoElement>(null)
  const player = useVideoPlayer(videoRef, playbackUrl)
  const fileInputRef = useRef<HTMLInputElement>(null)
  // Manual court marking applies to the video it was started for.
  const [markingFor, setMarkingFor] = useState<string | null>(null)
  const marking = source !== null && markingFor === source.key

  const openFilePicker = () => fileInputRef.current?.click()

  const onFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) selectFile(file)
    // Reset so choosing the same file again still fires a change event.
    e.target.value = ''
  }

  const { title } = headerTextFor(analysis.state, source?.name ?? null)
  const result = analysis.state.phase === 'ready' ? analysis.state.result : null
  const canMarkCourt =
    (analysis.state.phase === 'failed' && Boolean(analysis.state.jobId)) ||
    (result?.source === 'cv' && analysis.state.phase === 'ready')
  const startMarking =
    canMarkCourt && source && playbackUrl ? () => setMarkingFor(source.key) : undefined
  const downloadProgress = downloadProgressFor(source, analysis.state)

  return (
    <AppShell
      sidebar={<Sidebar backendStatus={backendStatus} />}
      header={
        <Header
          title={title}
          center={<ReviewSteps current={reviewStepFor(analysis.state)} />}
          actions={
            <Button
              variant={source ? 'secondary' : 'primary'}
              icon={<Upload size={16} />}
              aria-label={source ? 'Upload another rally' : 'Upload rally'}
              onClick={openFilePicker}
            >
              <span className={styles.uploadLabel}>
                {source ? 'Upload another' : 'Upload rally'}
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
          source={source}
          playbackUrl={playbackUrl}
          downloadProgress={downloadProgress}
          downloading={isAnalysisActive(analysis.state)}
          videoRef={videoRef}
          player={player}
          error={error}
          maxUploadBytes={config.maxUploadBytes}
          youtubeEnabled={config.youtube.enabled}
          onBrowse={openFilePicker}
          onFile={selectFile}
          onYouTube={selectYouTube}
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
