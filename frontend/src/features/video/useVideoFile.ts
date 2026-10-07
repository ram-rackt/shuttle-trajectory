import { useCallback, useEffect, useState } from 'react'
import { validateVideoFile } from './videoFile'

export interface LoadedVideo {
  file: File
  /** Object URL for local preview. Revoked automatically when replaced or cleared. */
  url: string
}

export interface VideoFileState {
  video: LoadedVideo | null
  error: string | null
  selectFile: (file: File) => void
  clear: () => void
  /** Called when the browser can't decode the selected file. */
  reportPlaybackError: () => void
}

/** Holds the locally selected rally video and its preview URL. */
export function useVideoFile(): VideoFileState {
  const [video, setVideo] = useState<LoadedVideo | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!video) return
    return () => URL.revokeObjectURL(video.url)
  }, [video])

  const selectFile = useCallback((file: File) => {
    const problem = validateVideoFile(file)
    if (problem) {
      setError(problem)
      return
    }
    setError(null)
    setVideo({ file, url: URL.createObjectURL(file) })
  }, [])

  const clear = useCallback(() => {
    setVideo(null)
    setError(null)
  }, [])

  const reportPlaybackError = useCallback(() => {
    if (!video) return
    setError(`“${video.file.name}” can’t be played in this browser. Try an MP4 (H.264) export.`)
    setVideo(null)
  }, [video])

  return { video, error, selectFile, clear, reportPlaybackError }
}
