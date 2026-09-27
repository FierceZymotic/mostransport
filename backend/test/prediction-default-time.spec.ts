import { describe, expect, it } from 'vitest';

import {
  DEFAULT_DEMO_PREDICTION_TIME,
  normalizePredictionTime,
  normalizeTelemetryTimestamp,
} from '../src/prediction/prediction.service.js';
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

  it('normalizes live runtime timestamps to the frozen demo contract', () => {
    const normalized = normalizePredictionTime(new Date('2026-09-27T09:42:44.000Z'));

    expect(normalized.toISOString()).toBe(DEFAULT_DEMO_PREDICTION_TIME.toISOString());
  });

  it('normalizes live telemetry timestamps to the frozen demo contract', () => {
    const normalized = normalizeTelemetryTimestamp(1790503305);

    expect(normalized).toBe(Math.floor(DEFAULT_DEMO_PREDICTION_TIME.getTime() / 1000));
  });
});
