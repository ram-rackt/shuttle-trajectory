import { useEffect, useState } from 'react'
import { DEFAULT_SERVER_CONFIG, getServerConfig, type ServerConfig } from '../api/client'

/**
 * The server's limits (upload size, video length, YouTube support), fetched
 * once. Until they arrive, or if the server can't be reached, the defaults
 * apply; the server enforces its own limits either way.
 */
export function useServerConfig(): ServerConfig {
  const [config, setConfig] = useState<ServerConfig>(DEFAULT_SERVER_CONFIG)

  useEffect(() => {
    const controller = new AbortController()
    getServerConfig(controller.signal)
      .then((raw) => setConfig(normalizeConfig(raw)))
      .catch(() => {})
    return () => controller.abort()
  }, [])

  return config
}

/**
 * Accepts only well-formed values, field by field, so an older or mismatched
 * server can never break the UI (missing fields keep their defaults).
 */
export function normalizeConfig(raw: unknown): ServerConfig {
  const d = DEFAULT_SERVER_CONFIG
  const obj = (v: unknown): Record<string, unknown> =>
    v && typeof v === 'object' ? (v as Record<string, unknown>) : {}
  const positive = (v: unknown, fallback: number) =>
    typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : fallback
  const r = obj(raw)
  const yt = obj(r.youtube)
  return {
    maxUploadBytes: positive(r.maxUploadBytes, d.maxUploadBytes),
    maxVideoSeconds: positive(r.maxVideoSeconds, d.maxVideoSeconds),
    youtube: {
      enabled: typeof yt.enabled === 'boolean' ? yt.enabled : d.youtube.enabled,
      maxSourceSeconds: positive(yt.maxSourceSeconds, d.youtube.maxSourceSeconds),
    },
  }
}
