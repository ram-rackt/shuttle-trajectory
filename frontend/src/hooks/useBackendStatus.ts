import { useEffect, useState } from 'react'
import { getHealth } from '../api/client'

export type BackendStatus = 'checking' | 'online' | 'offline'

/** One-shot health check against the API, run on mount. */
export function useBackendStatus(): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>('checking')

  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal)
      .then(() => setStatus('online'))
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          console.warn(err)
          setStatus('offline')
        }
      })
    return () => controller.abort()
  }, [])

  return status
}
