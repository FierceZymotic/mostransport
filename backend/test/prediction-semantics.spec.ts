import { describe, expect, it } from 'vitest';

import {
  classifyTargetActions,
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

  it('refuses a tie between different stops at the earliest time (no id-order choice)', () => {
    const at = (id: string, time: string, lat: number, lon: number, manual = false): TargetScheduleAction => ({
      target_action_id: id, target_time_begin: time, time_fact_begin: null, target_lat: lat, target_lon: lon, manual_fill: manual,
    });
    const T = new Date('2026-01-06T02:35:00.000Z');
    // Real Group-A tie from the organizer train plan: same minute, stops ~245 m apart.
    const differentGeometry = [at('53699456166', '2026-01-06T02:47:00.000Z', 55.6594982, 37.48110251), at('53699456165', '2026-01-06T02:47:00.000Z', 55.66166753, 37.48176323)];
    expect(classifyTargetActions(differentGeometry, T)).toEqual({ status: 'ambiguous', action: null, candidates: 2 });
    expect(resolveTargetActionFromRows(differentGeometry, T)).toBeNull();
    // Same place but different manual_fill is also a different model input.
    const differentManualFill = [at('1', '2026-01-06T02:47:00.000Z', 55.7, 37.6, true), at('2', '2026-01-06T02:47:00.000Z', 55.7, 37.6, false)];
    expect(classifyTargetActions(differentManualFill, T).status).toBe('ambiguous');
    // A later action does not break the tie, and a unique earliest action is chosen.
    const unique = [...differentGeometry.slice(0, 1), at('9', '2026-01-06T02:49:00.000Z', 55.7, 37.6)];
    expect(classifyTargetActions(unique, T)).toMatchObject({ status: 'ok', candidates: 1, action: { target_action_id: '53699456166' } });
    expect(classifyTargetActions([], T)).toEqual({ status: 'none', action: null, candidates: 0 });
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
