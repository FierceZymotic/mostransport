import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { classifyTargetActions } from '../src/prediction/schedule.repository.js';

// Shared with the Python side: tests/test_backend_conformance_fixtures.py reads the same file.
const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(resolve(here, '../../tests/fixtures/backend_ml_conformance.json'), 'utf8'));

describe('Backend <-> ML shared conformance fixtures', () => {
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
});
