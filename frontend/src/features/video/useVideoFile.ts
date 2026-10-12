import { useCallback, useEffect, useState } from 'react'
import type { YouTubeInfo } from '../../api/client'
import { validateVideoFile } from './videoFile'

/** A video chosen from the user's computer, previewed locally while it uploads. */
export interface FileSource {
  kind: 'file'
  /** Identifies this choice; analysis state is tagged with it. */
  key: string
  file: File
  /** Object URL for local preview. Revoked automatically when replaced or cleared. */
  url: string
  name: string
}

/** A YouTube video. The server downloads it; playback uses the server's copy. */
export interface YouTubeSource {
  kind: 'youtube'
  key: string
  info: YouTubeInfo
  /** The part to analyse, or null for the whole video. */
  startSec: number | null
  endSec: number | null
  name: string
}

export type VideoSource = FileSource | YouTubeSource

export interface VideoSourceState {
  source: VideoSource | null
  error: string | null
  selectFile: (file: File) => void
  selectYouTube: (info: YouTubeInfo, startSec: number | null, endSec: number | null) => void
  clear: () => void
  /** Called when the browser can't decode the selected video. */
  reportPlaybackError: () => void
}

let nextKey = 0

/** The video being reviewed: a local file or a YouTube video. */
export function useVideoSource(maxUploadBytes: number): VideoSourceState {
  const [source, setSource] = useState<VideoSource | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (source?.kind !== 'file') return
    return () => URL.revokeObjectURL(source.url)
  }, [source])

  const selectFile = useCallback(
    (file: File) => {
      const problem = validateVideoFile(file, maxUploadBytes)
      if (problem) {
        setError(problem)
        return
      }
      setError(null)
      const url = URL.createObjectURL(file)
      setSource({ kind: 'file', key: url, file, url, name: file.name })
    },
    [maxUploadBytes],
  )

  const selectYouTube = useCallback(
    (info: YouTubeInfo, startSec: number | null, endSec: number | null) => {
      setError(null)
      nextKey += 1
      setSource({
        kind: 'youtube',
        key: `youtube:${info.id}:${startSec ?? ''}:${endSec ?? ''}:${nextKey}`,
        info,
        startSec,
        endSec,
        name: info.title,
      })
    },
    [],
  )

  const clear = useCallback(() => {
    setSource(null)
    setError(null)
  }, [])

  const reportPlaybackError = useCallback(() => {
    if (!source) return
    setError(
      source.kind === 'file'
        ? `“${source.name}” can’t be played in this browser. Try an MP4 (H.264) export.`
        : `The downloaded copy of “${source.name}” can’t be played in this browser.`,
    )
    setSource(null)
  }, [source])

  return { source, error, selectFile, selectYouTube, clear, reportPlaybackError }
}
