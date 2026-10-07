import { useCallback, useEffect, useState, type RefObject } from 'react'

export interface PlayerState {
  currentTime: number
  duration: number
  paused: boolean
  muted: boolean
  /** Native frame size in pixels (0 until metadata loads). */
  videoWidth: number
  videoHeight: number
}

export interface PlayerControls {
  togglePlay: () => void
  seek: (time: number) => void
  seekBy: (delta: number) => void
  toggleMute: () => void
  enterFullscreen: () => void
}

export type VideoPlayer = PlayerState & PlayerControls

const INITIAL_STATE: PlayerState = {
  currentTime: 0,
  duration: 0,
  paused: true,
  muted: false,
  videoWidth: 0,
  videoHeight: 0,
}

const MEDIA_EVENTS = [
  'loadedmetadata',
  'durationchange',
  'timeupdate',
  'play',
  'pause',
  'ended',
  'seeked',
  'volumechange',
  'emptied',
] as const

function readState(v: HTMLVideoElement): PlayerState {
  return {
    currentTime: v.currentTime,
    duration: Number.isFinite(v.duration) ? v.duration : 0,
    paused: v.paused,
    muted: v.muted,
    videoWidth: v.videoWidth,
    videoHeight: v.videoHeight,
  }
}

/**
 * Mirrors a <video> element's playback state into React and exposes controls.
 * While playing, time is sampled every animation frame (timeupdate alone fires
 * only ~4×/s), which keeps the scrubber smooth and is what Phase 5 sync needs.
 */
export function useVideoPlayer(
  videoRef: RefObject<HTMLVideoElement | null>,
  src: string | null,
): VideoPlayer {
  const [state, setState] = useState<PlayerState>(INITIAL_STATE)

  useEffect(() => {
    const v = videoRef.current
    if (!v || !src) {
      setState(INITIAL_STATE)
      return
    }

    let frame = 0
    const tick = () => {
      setState(readState(v))
      frame = requestAnimationFrame(tick)
    }
    const sync = (event?: Event) => {
      setState(readState(v))
      if (event?.type === 'play') {
        cancelAnimationFrame(frame)
        frame = requestAnimationFrame(tick)
      } else if (event?.type === 'pause' || event?.type === 'ended' || event?.type === 'emptied') {
        cancelAnimationFrame(frame)
      }
    }

    MEDIA_EVENTS.forEach((e) => v.addEventListener(e, sync))
    sync()
    return () => {
      cancelAnimationFrame(frame)
      MEDIA_EVENTS.forEach((e) => v.removeEventListener(e, sync))
    }
  }, [videoRef, src])

  const togglePlay = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    if (v.paused || v.ended) {
      // play() rejects if interrupted by a pause or source change; that's expected.
      void v.play()?.catch(() => {})
    } else {
      v.pause()
    }
  }, [videoRef])

  const seek = useCallback(
    (time: number) => {
      const v = videoRef.current
      if (!v) return
      const max = Number.isFinite(v.duration) ? v.duration : 0
      v.currentTime = Math.min(Math.max(time, 0), max)
      setState(readState(v))
    },
    [videoRef],
  )

  const seekBy = useCallback(
    (delta: number) => {
      const v = videoRef.current
      if (v) seek(v.currentTime + delta)
    },
    [videoRef, seek],
  )

  const toggleMute = useCallback(() => {
    const v = videoRef.current
    if (v) v.muted = !v.muted
  }, [videoRef])

  const enterFullscreen = useCallback(() => {
    const v = videoRef.current as
      | (HTMLVideoElement & { webkitEnterFullscreen?: () => void })
      | null
    if (!v) return
    if (v.requestFullscreen) {
      void v.requestFullscreen().catch(() => {})
    } else {
      v.webkitEnterFullscreen?.() // iOS Safari
    }
  }, [videoRef])

  return { ...state, togglePlay, seek, seekBy, toggleMute, enterFullscreen }
}
