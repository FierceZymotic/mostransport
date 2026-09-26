import { Injectable } from '@nestjs/common';
import { Prisma } from '../generated/prisma/client.js';
import { PrismaService } from '../prisma/prisma.service.js';

export interface TargetScheduleAction {
  target_action_id: string;
  target_time_begin: string;
  time_fact_begin: string | null;
  target_lat: number;
  target_lon: number;
  manual_fill: boolean;
}

@Injectable()
export class ScheduleRepository {
  constructor(private readonly prisma: PrismaService) {}

 async findTargetAction(
  trId: string,
  predictionTime: Date,
): Promise<TargetScheduleAction | null> {
  const from = new Date(
    predictionTime.getTime() + 10 * 60 * 1000,
  );

  const to = new Date(
    predictionTime.getTime() + 15 * 60 * 1000,
  );

const rows = await this.prisma.$queryRaw<TargetScheduleAction[]>(
  Prisma.sql`
    SELECT
  tt_action_item_id AS target_action_id,
  time_begin::text AS target_time_begin,
  time_fact_begin::text AS time_fact_begin,
  ST_Y(geom) AS target_lat,
  ST_X(geom) AS target_lon,
  manual_fill
    FROM schedule_actions
    WHERE tr_id = ${trId}
      AND time_begin > ${from}
      AND time_begin <= ${to}
    ORDER BY time_begin ASC
    LIMIT 1
  `,
);

  return rows[0] ?? null;
}
async getCurrentDeviation(
  trId: string,
  predictionTime: Date,
): Promise<number> {
  const rows = await this.prisma.$queryRaw<
    { deviation_seconds: number | null }[]
  >(
    Prisma.sql`
      SELECT
        EXTRACT(
          EPOCH FROM (time_fact_begin - time_begin)
        ) AS deviation_seconds
      FROM schedule_actions
      WHERE tr_id = ${trId}
        AND time_begin <= ${predictionTime}
        AND time_fact_begin IS NOT NULL
        AND time_fact_begin <= ${predictionTime}
      ORDER BY time_begin DESC
      LIMIT 1
    `,
  );

  return rows[0]?.deviation_seconds ?? 0;
}
}