import { validateVideoFile } from './videoFile'

const MAX_VIDEO_BYTES = 2 * 1024 ** 3

function fakeFile(name: string, type: string, size = 1024): File {
  const file = new File(['x'], name, { type })
  Object.defineProperty(file, 'size', { value: size })
  return file
}

describe('validateVideoFile', () => {
  it.each([
    ['rally.mp4', 'video/mp4'],
    ['rally.mov', 'video/quicktime'],
    ['rally.webm', 'video/webm'],
    ['RALLY.MP4', ''], // empty MIME type, extension fallback
  ])('accepts %s (%s)', (name, type) => {
    expect(validateVideoFile(fakeFile(name, type), MAX_VIDEO_BYTES)).toBeNull()
  })

  it('rejects non-video files', () => {
    expect(validateVideoFile(fakeFile('notes.pdf', 'application/pdf'), MAX_VIDEO_BYTES)).toMatch(/isn’t a supported/)
  })

  it('rejects unsupported video formats', () => {
    expect(validateVideoFile(fakeFile('clip.avi', 'video/x-msvideo'), MAX_VIDEO_BYTES)).toMatch(/isn’t a supported/)
  })

  it('rejects empty files', () => {
    expect(validateVideoFile(fakeFile('rally.mp4', 'video/mp4', 0), MAX_VIDEO_BYTES)).toMatch(/empty/)
  })

  it('rejects files over the size limit', () => {
    expect(validateVideoFile(fakeFile('rally.mp4', 'video/mp4', MAX_VIDEO_BYTES + 1), MAX_VIDEO_BYTES)).toMatch(
      /limit/,
    )
  })
})
