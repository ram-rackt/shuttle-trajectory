import { AlertCircle, ExternalLink, Link2, Loader2 } from 'lucide-react'
import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { ApiError, getYouTubeInfo, type YouTubeInfo } from '../../api/client'
import { Button } from '../../components/ui/Button'
import { formatTime, parseTimecode } from '../../lib/format'
import { rangeProblem } from './youtubeRange'
import styles from './YouTubeForm.module.css'

interface YouTubeFormProps {
  onSubmit: (info: YouTubeInfo, startSec: number | null, endSec: number | null) => void
}

type Check =
  | { state: 'idle' }
  | { state: 'checking' }
  | { state: 'error'; message: string }
  | { state: 'ready'; info: YouTubeInfo }

/**
 * Paste a YouTube link, check it with the server (title, length, whether it
 * can be analysed), optionally choose a part, then analyse. Videos longer
 * than the server's limit must have a part chosen.
 */
export function YouTubeForm({ onSubmit }: YouTubeFormProps) {
  const [url, setUrl] = useState('')
  const [check, setCheck] = useState<Check>({ state: 'idle' })
  const controllerRef = useRef<AbortController | null>(null)
  const inputId = useId()
  const errorId = useId()

  useEffect(() => () => controllerRef.current?.abort(), [])

  const runCheck = async (e: FormEvent) => {
    e.preventDefault()
    if (!url.trim()) {
      setCheck({ state: 'error', message: 'Paste a YouTube link.' })
      return
    }
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setCheck({ state: 'checking' })
    try {
      const info = await getYouTubeInfo(url.trim(), controller.signal)
      setCheck({ state: 'ready', info })
    } catch (err) {
      if (controller.signal.aborted) return
      setCheck({
        state: 'error',
        message:
          err instanceof ApiError
            ? err.message
            : 'Couldn’t reach the analysis server. Check that the backend is running.',
      })
    }
  }

  const failed = check.state === 'error'
  return (
    <div className={styles.form}>
      <form className={styles.linkRow} onSubmit={runCheck} noValidate>
        <label htmlFor={inputId} className="visually-hidden">
          Link to a YouTube video
        </label>
        <span className={styles.inputWrap} data-invalid={failed || undefined}>
          <Link2 size={16} aria-hidden="true" />
          <input
            id={inputId}
            className={styles.input}
            type="url"
            inputMode="url"
            placeholder="https://www.youtube.com/watch?v=…"
            value={url}
            onChange={(e) => {
              setUrl(e.target.value)
              if (check.state !== 'checking') setCheck({ state: 'idle' })
            }}
            aria-invalid={failed || undefined}
            aria-describedby={failed ? errorId : undefined}
            autoComplete="off"
            spellCheck={false}
          />
        </span>
        <Button
          type="submit"
          variant={check.state === 'ready' ? 'secondary' : 'primary'}
          disabled={check.state === 'checking'}
          icon={check.state === 'checking' ? <Loader2 size={16} className={styles.spin} /> : undefined}
        >
          {check.state === 'checking' ? 'Checking…' : 'Check link'}
        </Button>
      </form>

      {failed && (
        <p id={errorId} className={styles.error} role="alert">
          <AlertCircle size={16} aria-hidden="true" />
          {check.message}
        </p>
      )}
      {check.state === 'ready' && (
        <VideoCard key={check.info.id} info={check.info} onSubmit={onSubmit} />
      )}
      {check.state === 'idle' && (
        <p className={styles.hint}>
          A broadcast filmed from behind the baseline works best. The video is downloaded to the
          analysis server first.
        </p>
      )}
    </div>
  )
}

function VideoCard({
  info,
  onSubmit,
}: {
  info: YouTubeInfo
  onSubmit: YouTubeFormProps['onSubmit']
}) {
  const limit = info.maxVideoSeconds
  const [partOnly, setPartOnly] = useState(info.needsRange)
  const [start, setStart] = useState('0:00')
  const [end, setEnd] = useState(formatTime(Math.min(info.durationSec, limit)))
  const startId = useId()
  const endId = useId()

  const startSec = parseTimecode(start)
  const endSec = parseTimecode(end)
  const problem = partOnly ? rangeProblem(startSec, endSec, info.durationSec, limit) : null

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (problem) return
    if (partOnly) onSubmit(info, startSec, endSec)
    else onSubmit(info, null, null)
  }

  return (
    <form className={styles.card} onSubmit={submit} aria-label="YouTube video">
      <div className={styles.summary}>
        {info.thumbnailUrl ? (
          <img className={styles.thumb} src={info.thumbnailUrl} alt="" loading="lazy" />
        ) : (
          <span className={styles.thumb} aria-hidden="true" />
        )}
        <div className={styles.meta}>
          <p className={styles.title} title={info.title}>
            {info.title}
          </p>
          <p className={styles.details}>
            {[info.channel, formatTime(info.durationSec), info.height ? `${info.height}p` : null]
              .filter(Boolean)
              .join(' · ')}
          </p>
          <a className={styles.external} href={info.url} target="_blank" rel="noreferrer">
            Open on YouTube <ExternalLink size={12} aria-hidden="true" />
          </a>
        </div>
      </div>

      <fieldset className={styles.range}>
        <legend className="visually-hidden">What to analyse</legend>
        <label className={styles.choice}>
          <input
            type="radio"
            name="part"
            checked={!partOnly}
            onChange={() => setPartOnly(false)}
            disabled={info.needsRange}
          />
          Whole video
        </label>
        <label className={styles.choice}>
          <input type="radio" name="part" checked={partOnly} onChange={() => setPartOnly(true)} />
          Part of it
        </label>
        {info.needsRange && (
          <p className={styles.note}>
            This video is longer than {formatTime(limit)}: choose up to {formatTime(limit)} of it.
          </p>
        )}
        {partOnly && (
          <div className={styles.times}>
            <label htmlFor={startId}>From</label>
            <input
              id={startId}
              className={styles.time}
              value={start}
              onChange={(e) => setStart(e.target.value)}
              placeholder="0:00"
              aria-invalid={startSec === null || undefined}
            />
            <label htmlFor={endId}>to</label>
            <input
              id={endId}
              className={styles.time}
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              placeholder={formatTime(limit)}
              aria-invalid={endSec === null || undefined}
            />
          </div>
        )}
        {problem && (
          <p className={styles.error} role="alert">
            <AlertCircle size={16} aria-hidden="true" />
            {problem}
          </p>
        )}
      </fieldset>

      <Button type="submit" variant="primary" disabled={Boolean(problem)}>
        Analyse video
      </Button>
    </form>
  )
}
