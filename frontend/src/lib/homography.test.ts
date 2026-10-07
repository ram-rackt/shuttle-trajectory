import { applyHomography, containRect } from './homography'

describe('applyHomography', () => {
  it('maps through an affine homography', () => {
    const H = [2, 0, 1, 0, 3, 2, 0, 0, 1]
    expect(applyHomography(H, 1, 1)).toEqual([3, 5])
  })

  it('divides by w for projective homographies', () => {
    const H = [1, 0, 0, 0, 1, 0, 0, 1, 1]
    expect(applyHomography(H, 2, 1)).toEqual([1, 0.5])
  })

  it('returns null for points behind the camera', () => {
    const H = [1, 0, 0, 0, 1, 0, 0, -1, 0.5]
    expect(applyHomography(H, 0, 1)).toBeNull()
  })
})

describe('containRect', () => {
  it('letterboxes a wide video in a tall box', () => {
    const r = containRect({ left: 0, top: 0, width: 160, height: 160 }, 1920, 1080)
    expect(r).toEqual({ left: 0, top: 35, width: 160, height: 90 })
  })

  it('pillarboxes a tall video in a wide box', () => {
    const r = containRect({ left: 10, top: 0, width: 400, height: 100 }, 1080, 1920)
    expect(r.width).toBeCloseTo(56.25)
    expect(r.left).toBeCloseTo(10 + (400 - 56.25) / 2)
  })
})
