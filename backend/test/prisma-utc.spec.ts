import { describe, expect, it, vi } from 'vitest';

import { createUtcPool } from '../src/prisma/prisma.service.js';

describe('Backend DB session time zone', () => {
  it('pins every new pooled connection to UTC before it is used', async () => {
    const pool = createUtcPool('postgresql://user@127.0.0.1:1/db');
    const client = { query: vi.fn(async () => ({})) };
    pool.emit('connect', client);
    expect(client.query).toHaveBeenCalledWith("SET TIME ZONE 'UTC'");
    await pool.end();
  });
});
