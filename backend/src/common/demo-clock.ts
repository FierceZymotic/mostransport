// One shared clock for the whole Backend session.
//
// The organizer schedule covers one historical service day, while a live NDTP emulator
// stamps packets with the current wall clock. For demos the Backend may translate *all*
// times by ONE constant offset, fixed once per session, so that live telemetry lands on
// the schedule day. A constant translation preserves every interval (Δt) and the packet
// order, including across midnight; the model's windows and lags depend on both.
// Nothing is ever mapped to a fixed instant, and dates are never replaced per packet.
//
// The same offset is applied to telemetry event time (at ingestion) and to "now", so the
// prediction time T, the history cutoff, trip matching and schedule lookup all share it.
//
//   DEMO_CLOCK_MODE=off        (default) times and "now" are used unchanged, e.g. for a
//                              replay of organizer data already on the schedule day
//   DEMO_CLOCK_MODE=day_shift  offset = whole days from the session's UTC start date to
//                              DEMO_CLOCK_TARGET_DAY (default 2026-01-06); the UTC time of
//                              day at session start is kept
//   DEMO_CLOCK_MODE=translate  offset = DEMO_CLOCK_ANCHOR (ISO-8601 with zone) - session start
export type DemoClockMode = 'off' | 'day_shift' | 'translate';

export const DEFAULT_DEMO_TARGET_DAY = '2026-01-06';

const DAY_MS = 24 * 60 * 60 * 1000;

function utcDayStart(value: Date): number {
  return Math.floor(value.getTime() / DAY_MS) * DAY_MS;
}

export interface DemoClockOptions {
  anchor?: Date;
  targetDay?: string;
}

export class DemoClock {
  readonly offsetSeconds: number;

  constructor(
    readonly mode: DemoClockMode,
    options: DemoClockOptions = {},
    sessionStart: Date = new Date(),
  ) {
    if (Number.isNaN(sessionStart.getTime())) {
      throw new Error('DemoClock requires a valid session start');
    }
    if (mode === 'translate') {
      const anchor = options.anchor;
      if (!anchor || Number.isNaN(anchor.getTime())) {
        throw new Error('DEMO_CLOCK_MODE=translate requires a valid DEMO_CLOCK_ANCHOR (ISO-8601 with time zone)');
      }
      this.offsetSeconds = Math.round((anchor.getTime() - sessionStart.getTime()) / 1000);
    } else if (mode === 'day_shift') {
      const day = options.targetDay ?? DEFAULT_DEMO_TARGET_DAY;
      if (!/^\d{4}-\d{2}-\d{2}$/.test(day) || Number.isNaN(Date.parse(`${day}T00:00:00Z`))) {
        throw new Error(`DEMO_CLOCK_TARGET_DAY must be YYYY-MM-DD, got "${day}"`);
      }
      this.offsetSeconds = (Date.parse(`${day}T00:00:00Z`) - utcDayStart(sessionStart)) / 1000;
    } else if (mode === 'off') {
      this.offsetSeconds = 0;
    } else {
      throw new Error(`Unsupported DEMO_CLOCK_MODE=${String(mode)}; expected "off", "day_shift" or "translate"`);
    }
  }

  static fromEnv(env: Record<string, string | undefined> = process.env, sessionStart: Date = new Date()): DemoClock {
    const mode = (env.DEMO_CLOCK_MODE ?? 'off').trim().toLowerCase() as DemoClockMode;
    const anchor = env.DEMO_CLOCK_ANCHOR ? new Date(env.DEMO_CLOCK_ANCHOR) : undefined;
    const targetDay = env.DEMO_CLOCK_TARGET_DAY?.trim() || undefined;
    return new DemoClock(mode, { anchor, targetDay }, sessionStart);
  }

  /** Device epoch seconds (NDTP timestamp) -> model-time epoch seconds. */
  toModelEpochSeconds(epochSeconds: number): number {
    return epochSeconds + this.offsetSeconds;
  }

  toModelDate(value: Date): Date {
    return new Date(value.getTime() + this.offsetSeconds * 1000);
  }

  /** Current wall clock in model time. */
  now(): Date {
    return this.toModelDate(new Date());
  }
}

/** Process-wide clock, fixed once per Backend session from the environment. */
export const demoClock = DemoClock.fromEnv();
