import {
  AlertCircle,
  CloudOff,
  ExternalLink,
  Eye,
  EyeOff,
  FileVideo,
  Gauge,
  Link2,
  Loader2,
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
  useId,
  useRef,
  useState,
  type CSSProperties,
  type DragEvent,
  type KeyboardEvent,
  type RefObject,
} from 'react'
import type { CourtCorners, YouTubeInfo } from '../../api/client'
import { Button } from '../../components/ui/Button'
import { Panel } from '../../components/ui/Panel'
import { formatBytes, formatRate, formatTime } from '../../lib/format'
import type { CameraCalibration, RallyAnalysis } from '../../types/analysis'
import { CourtMarker } from './CourtMarker'
import type { VideoSource, YouTubeSource } from './useVideoFile'
import { PLAYBACK_RATES, type VideoPlayer } from './useVideoPlayer'
import { VideoOverlay } from './VideoOverlay'
import styles from './VideoPanel.module.css'
import { YouTubeForm } from './YouTubeForm'

const SEEK_STEP_SECONDS = 5

interface VideoPanelProps {
  source: VideoSource | null
  /** What the player plays: the local file, or the server's copy of a YouTube video. */
  playbackUrl: string | null
  /** For a YouTube video still being fetched: download progress, 0 to 1 (null if unknown). */
  downloadProgress?: number | null
  /** False once the analysis has stopped (failed or cancelled): a YouTube video
   *  that isn't on the server by then won't arrive. */
  downloading?: boolean
  /** Owned by the parent so the court and timeline share the same clock. */
  videoRef: RefObject<HTMLVideoElement | null>
  player: VideoPlayer
  error: string | null
  maxUploadBytes: number
  youtubeEnabled: boolean
  onBrowse: () => void
  onFile: (file: File) => void
  onYouTube: (info: YouTubeInfo, startSec: number | null, endSec: number | null) => void
  onClear: () => void
  onPlaybackError: () => void
  /** CV results to draw over the video (court lines, players, shuttle). */
  overlay?: { analysis: RallyAnalysis; calibration: CameraCalibration } | null
  /** When set, the user is marking the court corners on the video. */
  marking?: { onSubmit: (corners: CourtCorners) => void; onCancel: () => void } | null
}

/** Left panel: choose a video (file or YouTube link), then play it with controls. */
export function VideoPanel({
  source,
  playbackUrl,
  downloadProgress = null,
  downloading = true,
  videoRef,
  player,
  error,
  maxUploadBytes,
  youtubeEnabled,
  onBrowse,
  onFile,
  onYouTube,
  onClear,
  onPlaybackError,
  overlay = null,
  marking = null,
}: VideoPanelProps) {
  const drag = useFileDrop(onFile)
  const [showOverlay, setShowOverlay] = useState(true)

  const meta = source ? (
    <>
      {overlay && playbackUrl && (
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
      {source.kind === 'youtube' ? (
        <a
          className={`${styles.fileName} ${styles.sourceLink}`}
          href={source.info.url}
          target="_blank"
          rel="noreferrer"
          title={`${source.name} (open on YouTube)`}
        >
          <span className={styles.sourceLinkText}>{source.name}</span>
          <ExternalLink size={12} aria-hidden="true" />
        </a>
      ) : (
        <>
          <span className={styles.fileName} title={source.name}>
            {source.name}
          </span>
          <span className={styles.fileSize}>{formatBytes(source.file.size)}</span>
        </>
      )}
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
      footer={<TransportBar player={player} disabled={!playbackUrl} />}
    >
      <div
        className={styles.stage}
        data-state={source ? 'loaded' : 'empty'}
        data-dragging={drag.isDragging || undefined}
        onDragEnter={drag.onDragEnter}
        onDragOver={drag.onDragOver}
        onDragLeave={drag.onDragLeave}
        onDrop={drag.onDrop}
        data-testid="video-stage"
      >
        {playbackUrl ? (
          <VideoSurface
            key={playbackUrl}
            src={playbackUrl}
            videoRef={videoRef}
            player={player}
            onError={onPlaybackError}
            overlay={showOverlay ? overlay : null}
            marking={marking}
          />
        ) : source?.kind === 'youtube' ? (
          <YouTubePending
            source={source}
            progress={downloadProgress}
            downloading={downloading}
          />
        ) : (
          <EmptyState
            error={error}
            maxUploadBytes={maxUploadBytes}
            youtubeEnabled={youtubeEnabled}
            onBrowse={onBrowse}
            onYouTube={onYouTube}
          />
        )}
        {drag.isDragging && (
          <div className={styles.dropOverlay} aria-hidden="true">
            <Upload size={28} />
            <span>{source ? 'Drop to replace video' : 'Drop to load video'}</span>
          </div>
        )}
      </div>
    </Panel>
  )
}

type SourceTab = 'file' | 'youtube'

const TABS: { id: SourceTab; label: string }[] = [
  { id: 'file', label: 'Upload a file' },
  { id: 'youtube', label: 'YouTube link' },
]

/** No video yet: upload a file, or (when the server allows it) paste a YouTube link. */
function EmptyState({
  error,
  maxUploadBytes,
  youtubeEnabled,
  onBrowse,
  onYouTube,
}: {
  error: string | null
  maxUploadBytes: number
  youtubeEnabled: boolean
  onBrowse: () => void
  onYouTube: VideoPanelProps['onYouTube']
}) {
  const [tab, setTab] = useState<SourceTab>('file')
  const tabsId = useId()
  const active: SourceTab = youtubeEnabled ? tab : 'file'

  return (
    <div className={styles.dropzone} data-tab={active}>
      {youtubeEnabled && (
        <div className={styles.tabs} role="tablist" aria-label="Video source">
          {TABS.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              role="tab"
              id={`${tabsId}-${id}`}
              aria-selected={active === id}
              aria-controls={`${tabsId}-${id}-panel`}
              className={styles.tab}
              onClick={() => setTab(id)}
            >
              {id === 'file' ? (
                <Upload size={15} aria-hidden="true" />
              ) : (
                <Link2 size={15} aria-hidden="true" />
              )}
              {label}
            </button>
          ))}
        </div>
      )}

      {active === 'file' ? (
        <div
          className={styles.tabPanel}
          role={youtubeEnabled ? 'tabpanel' : undefined}
          id={`${tabsId}-file-panel`}
          aria-labelledby={youtubeEnabled ? `${tabsId}-file` : undefined}
        >
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
          <p className={styles.dropHint}>
            MP4, MOV or WebM · up to {formatBytes(maxUploadBytes)}
          </p>
        </div>
      ) : (
        <div
          className={styles.tabPanel}
          role="tabpanel"
          id={`${tabsId}-youtube-panel`}
          aria-labelledby={`${tabsId}-youtube`}
        >
          <h3 className={styles.dropTitle}>Analyse a YouTube video</h3>
          <YouTubeForm onSubmit={onYouTube} />
        </div>
      )}
      {error && (
        <p className={styles.error} role="alert">
          <AlertCircle size={16} aria-hidden="true" />
          {error}
        </p>
      )}
    </div>
  )
}

/**
 * A YouTube video that isn't playable yet: its thumbnail and download progress
 * while the server fetches it, or a note when the analysis stopped before the
 * download finished (the court panel says why and offers a retry).
 */
function YouTubePending({
  source,
  progress,
  downloading,
}: {
  source: YouTubeSource
  progress: number | null
  downloading: boolean
}) {
  const { info, startSec, endSec } = source
  const part =
    startSec !== null && endSec !== null
      ? ` · analysing ${formatTime(startSec)}–${formatTime(endSec)}`
      : ''
  return (
    <div className={styles.pending} data-testid="youtube-pending">
      {info.thumbnailUrl && <img className={styles.pendingThumb} src={info.thumbnailUrl} alt="" />}
      {downloading ? (
        <div className={styles.pendingInfo} role="status">
          <Loader2 size={22} className={styles.spin} aria-hidden="true" />
          <p className={styles.pendingTitle}>Fetching the video from YouTube…</p>
          <p className={styles.pendingText}>
            {progress !== null ? `${Math.round(progress * 100)}% downloaded` : 'Starting'}
            {part}
          </p>
        </div>
      ) : (
        <div className={styles.pendingInfo}>
          <CloudOff size={22} aria-hidden="true" />
          <p className={styles.pendingTitle}>The video wasn’t downloaded</p>
          <p className={styles.pendingNote}>
            See the court view for what happened, then try again or choose another video.
          </p>
        </div>
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
      '<': () => player.stepPlaybackRate(-1),
      '>': () => player.stepPlaybackRate(1),
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
      aria-label="Video player. Space to play or pause, arrow keys to seek, < and > to change speed."
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
      {player.playbackRate !== 1 && (
        <span className={styles.rateBadge} aria-hidden="true" data-testid="rate-badge">
          {formatRate(player.playbackRate)}
        </span>
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
      <SpeedSelect player={player} disabled={disabled} />
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

/** Playback speed. The court and timeline follow the video's clock, so they slow down with it. */
function SpeedSelect({ player, disabled }: { player: VideoPlayer; disabled: boolean }) {
  const slowed = player.playbackRate !== 1
  return (
    <label className={styles.speed} data-active={slowed || undefined} title="Playback speed (< and >)">
      <Gauge size={15} aria-hidden="true" />
      <select
        className={styles.speedSelect}
        value={player.playbackRate}
        onChange={(e) => player.setPlaybackRate(Number(e.target.value))}
        disabled={disabled}
        aria-label="Playback speed"
      >
        {PLAYBACK_RATES.map((rate) => (
          <option key={rate} value={rate}>
            {formatRate(rate)}
          </option>
        ))}
      </select>
    </label>
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
