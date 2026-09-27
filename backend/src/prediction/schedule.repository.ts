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

export interface ScheduleFactRow {
  tt_action_item_id: string;
  tr_id: string;
  time_begin: string;
  time_fact_begin: string | null;
}

export function resolveTargetActionFromRows(
  rows: TargetScheduleAction[],
  predictionTime: Date,
): TargetScheduleAction | null {
  if (rows.length === 0) {
    return null;
  }

  const from = new Date(predictionTime.getTime() + 10 * 60 * 1000);
  const to = new Date(predictionTime.getTime() + 15 * 60 * 1000);

  const filtered = rows.filter((row) => {
    const time = new Date(row.target_time_begin);
    return time > from && time <= to;
  });

  if (filtered.length === 0) {
    return null;
  }

  filtered.sort((a, b) => {
    const timeDiff = new Date(a.target_time_begin).getTime() - new Date(b.target_time_begin).getTime();
    if (timeDiff !== 0) {
      return timeDiff;
    }

    const actionA = Number.parseInt(a.target_action_id, 10) || 0;
    const actionB = Number.parseInt(b.target_action_id, 10) || 0;
    return actionA - actionB;
  });

  return filtered[0];
}

export function resolveCurrentDeviationFromRows(
  rows: ScheduleFactRow[],
  predictionTime: Date,
): number {
  const eligible = rows.filter(
    (row) => row.time_fact_begin && new Date(row.time_fact_begin) <= predictionTime,
  );

  if (eligible.length === 0) {
    return 0;
  }

  const chosen = eligible.reduce((best, row) => {
    const factTime = new Date(row.time_fact_begin!);
    const bestFactTime = new Date(best.time_fact_begin!);

    if (factTime.getTime() > bestFactTime.getTime()) {
      return row;
    }

    if (factTime.getTime() === bestFactTime.getTime()) {
      const bestId = Number.parseInt(best.tt_action_item_id, 10) || 0;
      const currentId = Number.parseInt(row.tt_action_item_id, 10) || 0;
      if (currentId > bestId) {
        return row;
      }
    }

    return best;
  });

  const planned = new Date(chosen.time_begin);
  const factual = new Date(chosen.time_fact_begin!);
  return (factual.getTime() - planned.getTime()) / 1000;
}

@Injectable()
export class ScheduleRepository {
  constructor(private readonly prisma: PrismaService) {}

  async findTargetAction(
    trId: string,
    predictionTime: Date,
  ): Promise<TargetScheduleAction | null> {
    const from = new Date(predictionTime.getTime() + 10 * 60 * 1000);
    const to = new Date(predictionTime.getTime() + 15 * 60 * 1000);

    const rows = await this.prisma.$queryRaw<TargetScheduleAction[]>(
      Prisma.sql`
        SELECT
          tt_action_item_id AS target_action_id,
          time_begin::text AS target_time_begin,
          time_fact_begin::text AS time_fact_begin,
          ST_Y(geom) AS target_lat,
          ST_X(geom) AS target_lon,
          COALESCE(manual_fill::boolean, false) AS manual_fill
        FROM schedule_actions
        WHERE tr_id = ${trId}
          AND time_begin > ${from}
          AND time_begin <= ${to}
        ORDER BY time_begin ASC, CAST(tt_action_item_id AS bigint) ASC
      `,
    );

    return resolveTargetActionFromRows(rows, predictionTime);
  }

  async getCurrentDeviation(
    trId: string,
    predictionTime: Date,
  ): Promise<number> {
    const rows = await this.prisma.$queryRaw<ScheduleFactRow[]>(
      Prisma.sql`
        SELECT
          tt_action_item_id,
          tr_id,
          time_begin::text AS time_begin,
          time_fact_begin::text AS time_fact_begin
        FROM schedule_actions
        WHERE tr_id = ${trId}
          AND time_fact_begin IS NOT NULL
          AND time_fact_begin <= ${predictionTime}
        ORDER BY time_begin ASC, CAST(tt_action_item_id AS bigint) ASC
      `,
    );

    return resolveCurrentDeviationFromRows(rows, predictionTime);
  }
}
