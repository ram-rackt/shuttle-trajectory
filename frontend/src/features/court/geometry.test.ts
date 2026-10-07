import { COURT, courtLines, NET_X, toPolylinePoints, toSvg } from './geometry'

describe('toSvg', () => {
  it('maps normalized corners to the court rectangle in meters', () => {
    expect(toSvg({ x: 0, y: 0 })).toEqual({ x: 0, y: 0 })
    expect(toSvg({ x: 1, y: 1 })).toEqual({ x: COURT.length, y: COURT.width })
  })

  it('puts the net (y = 0.5) at the middle of the length', () => {
    expect(toSvg({ x: 0.5, y: 0.5 }).x).toBe(NET_X)
  })

  it("puts Player A's left sideline (x = 0) at the top", () => {
    expect(toSvg({ x: 0, y: 0.25 }).y).toBe(0)
  })
})

describe('toPolylinePoints', () => {
  it('formats points for an SVG polyline', () => {
    expect(
      toPolylinePoints([
        { x: 0, y: 0 },
        { x: 0.5, y: 0.5 },
      ]),
    ).toBe('0.000,0.000 6.700,3.050')
  })
})

describe('courtLines', () => {
  const lines = courtLines()

  it('draws the short service lines 1.98 m either side of the net', () => {
    const verticalXs = lines.filter(([x1, , x2]) => x1 === x2).map(([x]) => +x.toFixed(2))
    expect(verticalXs).toEqual(expect.arrayContaining([4.72, 8.68]))
  })

  it('keeps every line inside the court', () => {
    for (const [x1, y1, x2, y2] of lines) {
      for (const x of [x1, x2]) expect(x).toBeGreaterThanOrEqual(0)
      for (const x of [x1, x2]) expect(x).toBeLessThanOrEqual(COURT.length)
      for (const y of [y1, y2]) expect(y).toBeGreaterThanOrEqual(0)
      for (const y of [y1, y2]) expect(y).toBeLessThanOrEqual(COURT.width)
    }
  })
})
