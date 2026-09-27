import { describe, expect, it } from 'vitest';

import { DemoClock } from '../src/common/demo-clock.js';
import { InvalidPredictionTimeError, requirePredictionTime } from '../src/prediction/prediction.service.js';
import {
  resolveTargetActionFromRows,
  type TargetScheduleAction,
} from '../src/prediction/schedule.repository.js';

describe('schedule mismatch guard', () => {
  it('rejects schedule actions outside the required 10–15 minute horizon', () => {
    const rows: TargetScheduleAction[] = [
      {
        target_action_id: 'A-1',
        target_time_begin: '2026-01-06T04:27:00.000Z',
        time_fact_begin: null,
        target_lat: 55.75,
        target_lon: 37.61,
        manual_fill: false,
      },
      {
        target_action_id: 'A-2',
        target_time_begin: '2026-01-06T04:28:00.000Z',
        time_fact_begin: null,
        target_lat: 55.75,
        target_lon: 37.61,
        manual_fill: false,
      },
    ];

    const predictionTime = new Date('2026-01-06T03:35:00.000Z');
    const action = resolveTargetActionFromRows(rows, predictionTime);

    expect(action).toBeNull();
  });

  // The previous two tests asserted that live times are replaced by the fixed demo instant
  // 2026-01-06T03:35:00Z. Upstream (e1accf9) already changed telemetry to a replace-date
  // mapping, so the telemetry test failed at the base; both mappings destroy intervals
  // (every packet on one instant, or a 24 h jump at midnight) and were removed. The tests
  // below assert the replacement contract: explicit times are kept, and live time is only
  // shifted by one session-level constant offset that preserves packet intervals.
  it('keeps an explicit live prediction time instead of substituting a fixed demo instant', () => {
    const live = new Date('2026-09-27T09:42:44.000Z');
    expect(requirePredictionTime(live).toISOString()).toBe(live.toISOString());
    expect(() => requirePredictionTime(new Date('not a date'))).toThrow(InvalidPredictionTimeError);
  });

  it('translates live telemetry by a constant offset that preserves packet intervals', () => {
    const clock = new DemoClock('day_shift', {}, new Date('2026-09-27T09:00:00Z'));
    const live = [1790503305, 1790503318, 1790503318, 1790503333];
    const adapted = live.map((t) => clock.toModelEpochSeconds(t));
    expect(new Set(adapted).size).toBe(3); // not collapsed (only the genuine duplicate stays equal)
    expect(adapted.slice(1).map((t, i) => t - adapted[i])).toEqual(live.slice(1).map((t, i) => t - live[i]));
  });
});
