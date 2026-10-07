import {
  AlertCircle,
  Eye,
  EyeOff,
  FileVideo,
  Maximize,
  Pause,
  Play,
  Upload,
  Video,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react'
import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type DragEvent,
  type KeyboardEvent,
  type RefObject,
} from 'react'
import type { CourtCorners } from '../../api/client'
import { Button } from '../../components/ui/Button'
import { Panel } from '../../components/ui/Panel'
import { formatBytes, formatTime } from '../../lib/format'
import type { CameraCalibration, RallyAnalysis } from '../../types/analysis'
import { CourtMarker } from './CourtMarker'
import type { LoadedVideo } from './useVideoFile'
import type { VideoPlayer } from './useVideoPlayer'
import { VideoOverlay } from './VideoOverlay'
import styles from './VideoPanel.module.css'

const SEEK_STEP_SECONDS = 5

interface VideoPanelProps {
  video: LoadedVideo | null
  /** Owned by the parent so the court and timeline share the same clock. */
  videoRef: RefObject<HTMLVideoElement | null>
  player: VideoPlayer
  error: string | null
  onBrowse: () => void
  onFile: (file: File) => void
  onClear: () => void
  onPlaybackError: () => void
  /** CV results to draw over the video (court lines, players, shuttle). */
  overlay?: { analysis: RallyAnalysis; calibration: CameraCalibration } | null
  /** When set, the user is marking the court corners on the video. */
  marking?: { onSubmit: (corners: CourtCorners) => void; onCancel: () => void } | null
}

/** Left panel: rally video drop target, preview and playback controls. */
export function VideoPanel({
  video,
  videoRef,
  player,
  error,
  onBrowse,
  onFile,
  onClear,
  onPlaybackError,
  overlay = null,
  marking = null,
}: VideoPanelProps) {
  const drag = useFileDrop(onFile)
  const [showOverlay, setShowOverlay] = useState(true)

  const meta = video ? (
    <>
      {overlay && (
        <Button
          variant="ghost"
          size="sm"
          icon={showOverlay ? <Eye size={15} /> : <EyeOff size={15} />}
          aria-pressed={showOverlay}
          title="Show what the analysis detected on the video"
          onClick={() => setShowOverlay((v) => !v)}
        >
          Overlay
        </Button>
      )}
      <span className={styles.fileName} title={video.file.name}>
        {video.file.name}
      </span>
      <span className={styles.fileSize}>{formatBytes(video.file.size)}</span>
      <Button variant="secondary" size="sm" onClick={onBrowse}>
        Replace
      </Button>
      <Button
        variant="ghost"
        size="sm"
        icon={<X size={16} />}
        aria-label="Remove video"
        title="Remove video"
        onClick={onClear}
      />
    </>
  ) : (
    'No video loaded'
  )

  return (
    <Panel
      title="Rally video"
      icon={<Video size={16} />}
      meta={meta}
      footer={<TransportBar player={player} disabled={!video} />}
    >
      <div
        className={styles.stage}
        data-state={video ? 'loaded' : 'empty'}
        data-dragging={drag.isDragging || undefined}
        onDragEnter={drag.onDragEnter}
        onDragOver={drag.onDragOver}
        onDragLeave={drag.onDragLeave}
        onDrop={drag.onDrop}
        data-testid="video-stage"
      >
        {video ? (
          <VideoSurface
            key={video.url}
            src={video.url}
            videoRef={videoRef}
            player={player}
            onError={onPlaybackError}
            overlay={showOverlay ? overlay : null}
            marking={marking}
          />
        ) : (
          <EmptyDropzone error={error} onBrowse={onBrowse} />
        )}
        {drag.isDragging && (
          <div className={styles.dropOverlay} aria-hidden="true">
            <Upload size={28} />
            <span>{video ? 'Drop to replace video' : 'Drop to load video'}</span>
          </div>
        )}
      </div>
    </Panel>
  )
}

function EmptyDropzone({ error, onBrowse }: { error: string | null; onBrowse: () => void }) {
  return (
    <div className={styles.dropzone}>
      <span className={styles.dropIcon} aria-hidden="true">
        <FileVideo size={26} strokeWidth={1.6} />
      </span>
      <h3 className={styles.dropTitle}>Drop a rally video here</h3>
      <p className={styles.dropText}>
        A single rally filmed from behind the baseline gives the best results.
      </p>
      <Button variant="primary" icon={<Upload size={16} />} onClick={onBrowse}>
        Choose video
      </Button>
      <p className={styles.dropHint}>MP4, MOV or WebM · up to 2 GB</p>
      {error && (
        <p className={styles.error} role="alert">
          <AlertCircle size={16} aria-hidden="true" />
          {error}
        </p>
      )}
    </div>
  )
}

interface VideoSurfaceProps {
  src: string
  videoRef: RefObject<HTMLVideoElement | null>
  player: VideoPlayer
  onError: () => void
  overlay: VideoPanelProps['overlay']
  marking: VideoPanelProps['marking']
}

function VideoSurface({ src, videoRef, player, onError, overlay, marking }: VideoSurfaceProps) {
  // Corners are marked on a still frame.
  const { paused, togglePlay } = player
  useEffect(() => {
    if (marking && !paused) togglePlay()
  }, [marking, paused, togglePlay])

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const actions: Record<string, () => void> = {
      ' ': player.togglePlay,
      k: player.togglePlay,
      ArrowLeft: () => player.seekBy(-SEEK_STEP_SECONDS),
      ArrowRight: () => player.seekBy(SEEK_STEP_SECONDS),
      m: player.toggleMute,
      f: player.enterFullscreen,
    }
    const action = actions[e.key]
    if (action) {
      e.preventDefault()
      action()
    }
  }

  return (
    <div
      className={styles.surface}
      tabIndex={0}
      onKeyDown={onKeyDown}
      aria-label="Video player. Space to play or pause, arrow keys to seek."
      role="group"
    >
      <video
        ref={videoRef}
        className={styles.video}
        src={src}
        preload="auto"
        playsInline
        onClick={player.togglePlay}
        onError={onError}
        data-testid="rally-video"
      />
      {overlay && !marking && (
        <VideoOverlay
          analysis={overlay.analysis}
          calibration={overlay.calibration}
          time={player.currentTime}
          videoWidth={player.videoWidth}
          videoHeight={player.videoHeight}
        />
      )}
      {marking && player.videoWidth > 0 && (
        <CourtMarker
          videoWidth={player.videoWidth}
          videoHeight={player.videoHeight}
          onSubmit={marking.onSubmit}
          onCancel={marking.onCancel}
        />
      )}
      {player.paused && !marking && (
        <button
          type="button"
          className={styles.bigPlay}
          onClick={player.togglePlay}
          aria-label="Play video"
          tabIndex={-1}
        >
          <Play size={28} fill="currentColor" />
        </button>
      )}
    </div>
  )
}

function TransportBar({
  player,
  disabled,
}: {
  player: VideoPlayer
  disabled: boolean
}) {
  const { currentTime, duration, paused, muted } = player
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0
  return (
    <div className={styles.transport} role="group" aria-label="Playback controls">
      <Button
        variant="ghost"
        size="sm"
        icon={paused ? <Play size={16} /> : <Pause size={16} />}
        aria-label={paused ? 'Play' : 'Pause'}
        onClick={player.togglePlay}
        disabled={disabled}
      />
      <span className={styles.time}>{formatTime(currentTime)}</span>
      <input
        type="range"
        className={styles.seek}
        min={0}
        max={duration || 0}
        step="any"
        value={Math.min(currentTime, duration || 0)}
        onChange={(e) => player.seek(Number(e.target.value))}
        disabled={disabled || duration === 0}
        aria-label="Seek"
        aria-valuetext={`${formatTime(currentTime)} of ${formatTime(duration)}`}
        style={{ '--progress': `${progress}%` } as CSSProperties}
      />
      <span className={styles.time}>{formatTime(duration)}</span>
      <Button
        variant="ghost"
        size="sm"
        icon={muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
        aria-label={muted ? 'Unmute' : 'Mute'}
        onClick={player.toggleMute}
        disabled={disabled}
      />
      <Button
        variant="ghost"
        size="sm"
        icon={<Maximize size={16} />}
        aria-label="Fullscreen"
        onClick={player.enterFullscreen}
        disabled={disabled}
      />
    </div>
  )
}

/**
 * Drag-and-drop handling for files. A depth counter avoids flicker when the
 * pointer moves over child elements (dragleave fires on each child boundary).
 */
function useFileDrop(onFile: (file: File) => void) {
  const depth = useRef(0)
  const [isDragging, setIsDragging] = useState(false)

  const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes('Files')

  return {
    isDragging,
    onDragEnter: (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      depth.current += 1
      setIsDragging(true)
    },
    onDragOver: (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    },
    onDragLeave: (e: DragEvent) => {
      if (!hasFiles(e)) return
      depth.current = Math.max(0, depth.current - 1)
      if (depth.current === 0) setIsDragging(false)
    },
    onDrop: (e: DragEvent) => {
      e.preventDefault()
      depth.current = 0
      setIsDragging(false)
      const file = e.dataTransfer?.files?.[0]
      if (file) onFile(file)
    },
  }
}
