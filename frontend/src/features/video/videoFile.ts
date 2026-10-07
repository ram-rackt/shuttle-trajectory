import { formatBytes } from '../../lib/format'

export const ACCEPTED_VIDEO_EXTENSIONS = ['mp4', 'm4v', 'mov', 'webm'] as const
const ACCEPTED_VIDEO_MIME_TYPES = ['video/mp4', 'video/x-m4v', 'video/quicktime', 'video/webm']

/** Value for the file input's `accept` attribute. */
export const VIDEO_ACCEPT_ATTR = [
  ...ACCEPTED_VIDEO_MIME_TYPES,
  ...ACCEPTED_VIDEO_EXTENSIONS.map((ext) => `.${ext}`),
].join(',')

export const MAX_VIDEO_BYTES = 2 * 1024 ** 3

function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot === -1 ? '' : name.slice(dot + 1).toLowerCase()
}

/**
 * Returns a user-facing error message, or null if the file looks like a
 * supported video. Browsers sometimes report an empty MIME type, so the
 * extension is accepted as a fallback.
 */
export function validateVideoFile(file: File): string | null {
  const typeOk = ACCEPTED_VIDEO_MIME_TYPES.includes(file.type)
  const extOk = (ACCEPTED_VIDEO_EXTENSIONS as readonly string[]).includes(extensionOf(file.name))
  if (!typeOk && !extOk) {
    return `“${file.name}” isn’t a supported video. Use MP4, MOV or WebM.`
  }
  if (file.size === 0) {
    return `“${file.name}” is empty.`
  }
  if (file.size > MAX_VIDEO_BYTES) {
    return `“${file.name}” is ${formatBytes(file.size)}. The limit is ${formatBytes(MAX_VIDEO_BYTES)}.`
  }
  return null
}
