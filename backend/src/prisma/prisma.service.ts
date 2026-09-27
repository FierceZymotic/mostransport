import 'dotenv/config';
import { Injectable } from '@nestjs/common';
import { PrismaPg } from '@prisma/adapter-pg';
import pg from 'pg';
import { PrismaClient } from '../generated/prisma/client.js';

/**
 * @prisma/adapter-pg binds a JS Date as a timestamp string without an offset, which
 * Postgres reads in the *session* TimeZone. Every prediction query compares timestamptz
 * columns with a Date (T, the history window, the target horizon, facts <= T) and
 * telemetry is written as Dates, so each session is pinned to UTC. Without this a non-UTC
 * server (e.g. TimeZone = Europe/Moscow) shifts every bound by the zone offset.
 */
export function createUtcPool(connectionString: string): pg.Pool {
  const pool = new pg.Pool({ connectionString });
  pool.on('connect', (client) => {
    client.query("SET TIME ZONE 'UTC'").catch((error: unknown) => {
      console.warn(`Could not pin DB session TimeZone to UTC: ${error instanceof Error ? error.message : String(error)}`);
    });
  });
  return pool;
}

@Injectable()
export class PrismaService extends PrismaClient {
  constructor() {
    const adapter = new PrismaPg(createUtcPool(process.env.DATABASE_URL!));

    super({ adapter });
  }
  async testConnection() {
    return this.vehicles.count();
}
}
