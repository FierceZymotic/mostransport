import { describe, expect, it } from 'vitest';

import { DemoClock } from '../src/common/demo-clock.js';

const s = (iso: string) => Math.floor(Date.parse(iso) / 1000);

/** Upstream e1accf9 live mapping, kept only to document why it was replaced. */
function replaceDateKeepTimeOfDay(epochSeconds: number): number {
  const d = new Date(epochSeconds * 1000);
  return Math.floor(Date.UTC(2026, 0, 6, d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds()) / 1000);
}

describe('DemoClock (one session-level constant offset)', () => {
  it('is the identity by default (DEMO_CLOCK_MODE unset/off): replay data on the schedule day is unchanged', () => {
    const clock = DemoClock.fromEnv({}, new Date('2026-09-27T09:00:00Z'));
    expect(clock.mode).toBe('off');
    expect(clock.offsetSeconds).toBe(0);
    const t = s('2026-01-06T03:35:07Z');
    expect(clock.toModelEpochSeconds(t)).toBe(t);
  });

  it('day_shift lands the live time of day on the schedule day with a whole-day offset fixed at session start', () => {
    const clock = DemoClock.fromEnv({ DEMO_CLOCK_MODE: 'day_shift' }, new Date('2026-09-27T09:42:44Z'));
    expect(Number.isInteger(clock.offsetSeconds / 86400)).toBe(true);
    expect(new Date(clock.toModelEpochSeconds(s('2026-09-27T09:42:44Z')) * 1000).toISOString()).toBe('2026-01-06T09:42:44.000Z');
    const custom = DemoClock.fromEnv({ DEMO_CLOCK_MODE: 'day_shift', DEMO_CLOCK_TARGET_DAY: '2026-01-07' }, new Date('2026-09-27T09:42:44Z'));
    expect(custom.offsetSeconds - clock.offsetSeconds).toBe(86400);
  });

  it('translate maps the session start onto the configured anchor', () => {
    const start = new Date('2026-09-27T09:00:00Z');
    const clock = new DemoClock('translate', { anchor: new Date('2026-01-06T06:00:00Z') }, start);
    expect(clock.toModelDate(start).toISOString()).toBe('2026-01-06T06:00:00.000Z');
  });

  it('preserves adapt(t2) - adapt(t1) == t2 - t1 and the order, in every mode', () => {
    const start = new Date('2026-09-27T09:00:00Z');
    const clocks = [
      new DemoClock('off', {}, start),
      new DemoClock('day_shift', {}, start),
      new DemoClock('translate', { anchor: new Date('2026-01-06T06:00:00Z') }, start),
    ];
    const live = [s('2026-09-27T09:00:00Z'), 1790503305, 1790503318, 1790503318, 1790503333, s('2026-09-28T02:00:00Z')];
    for (const clock of clocks) {
      const adapted = live.map((t) => clock.toModelEpochSeconds(t));
      for (let i = 1; i < live.length; i++) {
        expect(adapted[i] - adapted[i - 1]).toBe(live[i] - live[i - 1]);
      }
      expect(new Set(adapted).size).toBe(new Set(live).size); // only the genuine duplicate stays equal
    }
  });

  it('survives midnight crossing (the replaced replace-date mapping jumped back 24 h)', () => {
    const clock = DemoClock.fromEnv({ DEMO_CLOCK_MODE: 'day_shift' }, new Date('2026-09-27T23:59:00Z'));
    const t1 = s('2026-09-27T23:59:55Z');
    const t2 = s('2026-09-28T00:00:05Z');
    expect(clock.toModelEpochSeconds(t2) - clock.toModelEpochSeconds(t1)).toBe(10);
    expect(new Date(clock.toModelEpochSeconds(t2) * 1000).toISOString()).toBe('2026-01-07T00:00:05.000Z');
    // Old behaviour: 00:00:05 of the next day became 2026-01-06 00:00:05, i.e. 23:59:50 BEFORE t1.
    expect(replaceDateKeepTimeOfDay(t2) - replaceDateKeepTimeOfDay(t1)).toBe(-86390);
  });

  it('uses the same offset for "now" as for ingested telemetry', () => {
    const clock = new DemoClock('day_shift', {}, new Date('2026-09-27T09:00:00Z'));
    const wall = new Date('2026-09-27T10:11:12Z');
    expect(clock.toModelDate(wall).getTime() / 1000).toBe(clock.toModelEpochSeconds(wall.getTime() / 1000));
  });

  it('rejects invalid configuration instead of guessing', () => {
    expect(() => DemoClock.fromEnv({ DEMO_CLOCK_MODE: 'translate' })).toThrow(/DEMO_CLOCK_ANCHOR/);
    expect(() => DemoClock.fromEnv({ DEMO_CLOCK_MODE: 'translate', DEMO_CLOCK_ANCHOR: 'nonsense' })).toThrow(/DEMO_CLOCK_ANCHOR/);
    expect(() => DemoClock.fromEnv({ DEMO_CLOCK_MODE: 'day_shift', DEMO_CLOCK_TARGET_DAY: '06.01.2026' })).toThrow(/YYYY-MM-DD/);
    expect(() => DemoClock.fromEnv({ DEMO_CLOCK_MODE: 'freeze' })).toThrow(/Unsupported/);
  });
});
