import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { classifyTargetActions, resolveCurrentDeviationFromRows } from '../src/prediction/schedule.repository.js';

// Shared with the Python side: tests/test_backend_conformance_fixtures.py reads the same file.
const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(resolve(here, '../../tests/fixtures/backend_ml_conformance.json'), 'utf8'));

describe('Backend <-> ML shared conformance fixtures', () => {
  for (const c of fixture.current_deviation) {
    it(`current deviation: ${c.name}`, () => {
      const rows = c.rows.map((r: any) => ({ tt_action_item_id: r.id, tr_id: 'X', time_begin: r.plan, time_fact_begin: r.fact }));
      expect(resolveCurrentDeviationFromRows(rows, new Date(c.T))).toBe(c.expected_seconds);
    });
  }
  for (const c of fixture.target_selection) {
    it(`target selection: ${c.name}`, () => {
      const rows = c.rows.map((r: any) => ({
        target_action_id: r.id, target_time_begin: r.plan, time_fact_begin: null, target_lat: r.lat, target_lon: r.lon, manual_fill: r.manual_fill,
      }));
      const out = classifyTargetActions(rows, new Date(c.T));
      expect(out.status).toBe(c.expected_status);
      expect(out.action?.target_action_id ?? null).toBe(c.expected_id);
    });
  }
  it('Postgres timestamptz text parses identically in any DB session time zone (offset is present)', () => {
    expect(new Date('2026-01-06 09:36:00+03').toISOString()).toBe('2026-01-06T06:36:00.000Z');
    expect(new Date('2026-01-06 06:36:00+00').toISOString()).toBe('2026-01-06T06:36:00.000Z');
    const utc = resolveCurrentDeviationFromRows([{ tt_action_item_id: '1', tr_id: 'X', time_begin: '2026-01-06 06:30:00+00', time_fact_begin: '2026-01-06 06:32:30.5+00' }], new Date('2026-01-06T07:00:00Z'));
    const msk = resolveCurrentDeviationFromRows([{ tt_action_item_id: '1', tr_id: 'X', time_begin: '2026-01-06 09:30:00+03', time_fact_begin: '2026-01-06 09:32:30.5+03' }], new Date('2026-01-06T07:00:00Z'));
    expect(utc).toBe(150.5);
    expect(msk).toBe(utc);
  });
});
