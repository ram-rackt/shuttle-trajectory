import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'

export interface PlayerState {
  currentTime: number
  duration: number
  paused: boolean
  muted: boolean
  /** Playback speed: 1 is normal, below 1 is slow motion. */
  playbackRate: number
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
  /** Sets the speed (one of PLAYBACK_RATES); kept when another video is loaded. */
  setPlaybackRate: (rate: number) => void
  /** Next slower (-1) or faster (+1) speed in PLAYBACK_RATES. */
  stepPlaybackRate: (direction: -1 | 1) => void
}

/** Speeds offered in the UI, slowest first. */
export const PLAYBACK_RATES = [0.25, 0.5, 0.75, 1, 1.5, 2] as const

export type VideoPlayer = PlayerState & PlayerControls

const INITIAL_STATE: PlayerState = {
  currentTime: 0,
  duration: 0,
  paused: true,
  muted: false,
  playbackRate: 1,
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
  'ratechange',
  'emptied',
] as const

function readState(v: HTMLVideoElement): PlayerState {
  return {
    currentTime: v.currentTime,
    duration: Number.isFinite(v.duration) ? v.duration : 0,
    paused: v.paused,
    muted: v.muted,
    playbackRate: v.playbackRate,
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
  // The speed the user chose. A new source resets the element to 1x, so it is
  // re-applied whenever a video is attached.
  const rateRef = useRef(1)

  useEffect(() => {
    const v = videoRef.current
    if (!v || !src) {
      setState({ ...INITIAL_STATE, playbackRate: rateRef.current })
      return
    }
    applyRate(v, rateRef.current)

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

    // Loading a source resets playbackRate to defaultPlaybackRate; keep ours.
    const keepRate = () => applyRate(v, rateRef.current)
    v.addEventListener('loadedmetadata', keepRate)
    MEDIA_EVENTS.forEach((e) => v.addEventListener(e, sync))
    sync()
    return () => {
      cancelAnimationFrame(frame)
      v.removeEventListener('loadedmetadata', keepRate)
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

  const setPlaybackRate = useCallback(
    (rate: number) => {
      if (!(PLAYBACK_RATES as readonly number[]).includes(rate)) return
      rateRef.current = rate
      const v = videoRef.current
      if (v) {
        applyRate(v, rate)
        setState(readState(v))
      } else {
        setState((s) => ({ ...s, playbackRate: rate }))
      }
    },
    [videoRef],
  )

  const stepPlaybackRate = useCallback(
    (direction: -1 | 1) => {
      const i = PLAYBACK_RATES.indexOf(rateRef.current as (typeof PLAYBACK_RATES)[number])
      const next = PLAYBACK_RATES[Math.min(Math.max(i + direction, 0), PLAYBACK_RATES.length - 1)]
      setPlaybackRate(next)
    },
    [setPlaybackRate],
  )

  return {
    ...state,
    togglePlay,
    seek,
    seekBy,
    toggleMute,
    enterFullscreen,
    setPlaybackRate,
    stepPlaybackRate,
  }
}

function applyRate(v: HTMLVideoElement, rate: number) {
  v.defaultPlaybackRate = rate
  v.playbackRate = rate
}
