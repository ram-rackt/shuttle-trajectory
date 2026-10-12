import { formatBytes, formatTime, parseTimecode } from './format'

describe('formatTime', () => {
  it.each([
    [0, '0:00'],
    [5.9, '0:05'],
    [65, '1:05'],
    [3600, '1:00:00'],
    [3725, '1:02:05'],
    [Number.NaN, '0:00'],
    [Number.POSITIVE_INFINITY, '0:00'],
    [-3, '0:00'],
  ])('formats %s as %s', (input, expected) => {
    expect(formatTime(input)).toBe(expected)
  })
})

describe('formatBytes', () => {
  it.each([
    [0, '0 B'],
    [512, '512 B'],
    [1536, '1.5 KB'],
    [25.5 * 1024 * 1024, '25.5 MB'],
    [300 * 1024 * 1024, '300 MB'],
    [2 * 1024 ** 3, '2.0 GB'],
  ])('formats %s as %s', (input, expected) => {
    expect(formatBytes(input)).toBe(expected)
  })
})

describe('parseTimecode', () => {
  it.each([
    ['83', 83],
    ['1:23', 83],
    ['01:02:03', 3723],
    ['0:05.5', 5.5],
    [' 10:00 ', 600],
  ])('parses %s', (text, seconds) => {
    expect(parseTimecode(text)).toBe(seconds)
  })

  it.each(['', 'abc', '1:75', '1::2', '-5', '1:2:3:4', '1:2x'])('rejects %j', (text) => {
    expect(parseTimecode(text)).toBeNull()
  })
})
