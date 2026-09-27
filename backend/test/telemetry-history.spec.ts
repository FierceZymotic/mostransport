import { describe, expect, it } from 'vitest';

import { selectContractHistory, type HistoryRow } from '../src/telemetry/telemetry.repository.js';

type Row = HistoryRow & { tag: string; speed: number };

const T = new Date('2026-01-06T12:00:00Z');
let nextId = 1n;
function row(offsetSeconds: number, valid = true, id?: bigint, speed = 0): Row {
  const rid = id ?? nextId++;
  return {
    id: rid,
    timestamp: new Date(T.getTime() + offsetSeconds * 1000),
    location_valid: valid,
    longitude: valid ? 37.6 : null,
    latitude: valid ? 55.7 : null,
    speed,
    tag: `${offsetSeconds}:${rid}`,
  };
}
const ids = (rows: Row[]) => rows.map((r) => r.id);

describe('selectContractHistory (Contract v1 §7)', () => {
  it('never admits packets after the prediction time', () => {
    const rows = [row(-60), row(0), row(1), row(600)];
    const out = selectContractHistory(rows, T);
    expect(out.every((r) => r.timestamp.getTime() <= T.getTime())).toBe(true);
    expect(out.map((r) => r.tag)).toEqual([rows[0].tag, rows[1].tag]);
  });

  it('includes a packet exactly at T and excludes one 1 ms after T', () => {
    const atT = row(0);
    const after = { ...row(0), id: 999n, timestamp: new Date(T.getTime() + 1), tag: 'after' };
    expect(ids(selectContractHistory([after, atT], T))).toEqual([atT.id]);
  });

  it('a later-inserted future packet does not change the history at T', () => {
    const base = [row(-120, true, 1n, 10), row(-60, false, 2n, 20), row(-10, true, 3n, 30)];
    const before = selectContractHistory(base, T);
    const after = selectContractHistory([...base, row(5, true, 4n, 99), row(900, false, 5n, 0)], T);
    expect(after).toEqual(before);
  });

  it('keeps equal-time packets with different speeds, validity and rows (no timestamp-keyed collapse)', () => {
    const a = row(-30, true, 101n, 12);
    const b = row(-30, false, 102n, 0);
    const c = row(-30, true, 103n, 14);
    const out = selectContractHistory([c, a, b], T);
    expect(ids(out)).toEqual([101n, 102n, 103n]);
    expect(out.map((r) => r.speed)).toEqual([12, 0, 14]);
    expect(out.map((r) => r.location_valid)).toEqual([true, false, true]);
  });

  it('orders deterministically by (timestamp, id) regardless of input order', () => {
    const rows = [row(-5, true, 7n), row(-100, true, 3n), row(-5, true, 2n), row(-50, true, 9n)];
    const shuffled = [rows[2], rows[0], rows[3], rows[1]];
    expect(ids(selectContractHistory(rows, T))).toEqual([3n, 9n, 2n, 7n]);
    expect(ids(selectContractHistory(shuffled, T))).toEqual([3n, 9n, 2n, 7n]);
  });

  it('uses the half-open window (T-15m, T] and adds last-packet / last-valid-GPS anchors once', () => {
    const oldValid = row(-3600, true);
    const atBoundary = row(-900, false); // exactly T-15m: outside the window
    const inWindowInvalid = row(-600, false);
    const out = selectContractHistory([oldValid, atBoundary, inWindowInvalid], T);
    // window: inWindowInvalid; last packet = inWindowInvalid; last strict-valid = oldValid (anchor older than window)
    expect(out.map((r) => r.tag)).toEqual([oldValid.tag, inWindowInvalid.tag]);
  });

  it('keeps an old last-packet anchor when the window is empty', () => {
    const older = row(-7200, false);
    const lastOld = row(-3600, false);
    expect(ids(selectContractHistory([older, lastOld], T))).toEqual([lastOld.id]);
  });

  it('recovers the latest strict-valid GPS when the latest packet is invalid', () => {
    const validBefore = row(-1200, true, 11n); // outside the window
    const invalidLatest = row(-5, false, 12n);
    const out = selectContractHistory([validBefore, invalidLatest], T);
    expect(ids(out)).toEqual([11n, 12n]);
    expect(out.filter((r) => r.location_valid).at(-1)?.id).toBe(11n);
  });

  it('a valid flag without coordinates is not strict-valid GPS', () => {
    const trulyValid = row(-2000, true, 21n);
    const flagOnly = { ...row(-1500, true, 22n), latitude: null, longitude: null };
    const latest = row(-5, false, 23n);
    expect(ids(selectContractHistory([trulyValid, flagOnly, latest], T))).toEqual([21n, 23n]);
  });

  it('returns an empty history when nothing is at or before T', () => {
    expect(selectContractHistory([row(10)], T)).toEqual([]);
  });
});
