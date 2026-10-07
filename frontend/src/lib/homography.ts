/** Applies a row-major 3x3 homography to a point. Null for points behind the camera. */
export function applyHomography(H: readonly number[], x: number, y: number): [number, number] | null {
  const w = H[6] * x + H[7] * y + H[8]
  if (!(w > 1e-9)) return null
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w]
}

/**
 * The displayed content box of a media element sized with `object-fit: contain`:
 * where the frame actually is inside the element, in client pixels.
 */
export function containRect(
  element: { left: number; top: number; width: number; height: number },
  mediaWidth: number,
  mediaHeight: number,
): { left: number; top: number; width: number; height: number } {
  if (mediaWidth <= 0 || mediaHeight <= 0) return element
  const scale = Math.min(element.width / mediaWidth, element.height / mediaHeight)
  const width = mediaWidth * scale
  const height = mediaHeight * scale
  return {
    left: element.left + (element.width - width) / 2,
    top: element.top + (element.height - height) / 2,
    width,
    height,
  }
}
