import { describe, expect, it } from 'vitest';

import {
  resolveCurrentDeviationFromRows,
  resolveTargetActionFromRows,
  type TargetScheduleAction,
} from '../src/prediction/schedule.repository.js';

describe('schedule semantics', () => {
  it('chooses the earliest target action in the horizon and breaks ties deterministically', () => {
    const rows: TargetScheduleAction[] = [
      {
        target_action_id: '200',
        target_time_begin: '2026-01-06T03:47:00.000Z',
        time_fact_begin: null,
        target_lat: 55.75,
        target_lon: 37.61,
        manual_fill: false,
      },
      {
        target_action_id: '100',
        target_time_begin: '2026-01-06T03:47:00.000Z',
        time_fact_begin: null,
        target_lat: 55.75,
        target_lon: 37.61,
        manual_fill: false,
      },
      {
        target_action_id: '300',
        target_time_begin: '2026-01-06T03:55:00.000Z',
        time_fact_begin: null,
        target_lat: 55.76,
        target_lon: 37.62,
        manual_fill: false,
      },
    ];

    const predictionTime = new Date('2026-01-06T03:35:00.000Z');
    const action = resolveTargetActionFromRows(rows, predictionTime);

    expect(action?.target_action_id).toBe('100');
  });

  it('uses the latest confirmed fact at or before T and ignores future facts', () => {
    const rows = [
      {
        tt_action_item_id: '10',
        tr_id: '131672',
        time_begin: '2026-01-06T03:35:00.000Z',
        time_fact_begin: '2026-01-06T03:36:00.000Z',
      },
      {
        tt_action_item_id: '9',
        tr_id: '131672',
        time_begin: '2026-01-06T03:30:00.000Z',
        time_fact_begin: '2026-01-06T03:50:00.000Z',
      },
      {
        tt_action_item_id: '11',
        tr_id: '131672',
        time_begin: '2026-01-06T03:32:00.000Z',
        time_fact_begin: '2026-01-06T03:33:00.000Z',
      },
    ];

    const deviation = resolveCurrentDeviationFromRows(rows, new Date('2026-01-06T03:35:00.000Z'));
    expect(deviation).toBe(60);
  });
});
