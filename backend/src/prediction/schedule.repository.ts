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

export type TargetSelectionStatus = 'ok' | 'none' | 'ambiguous';

export interface TargetSelection {
  status: TargetSelectionStatus;
  action: TargetScheduleAction | null;
  /** Number of planned actions at the earliest time in the horizon. */
  candidates: number;
}

function numericActionId(row: TargetScheduleAction): number {
  return Number.parseInt(row.target_action_id, 10) || 0;
}

/**
 * Contract docs §5.1: the target is the earliest planned action with
 * T + 10 min < time_begin <= T + 15 min. The organizer defines no tie-break when several
 * actions share that earliest time, so an id order must not decide which physical stop is
 * the target:
 *  - all tied actions identical in time, geometry and manual_fill (= identical model inputs,
 *    proven equivalent): collapse; the reported id is trace-only (lowest numeric id);
 *  - otherwise: AMBIGUOUS, no action (the caller refuses with 422 TARGET_AMBIGUOUS).
 */
export function classifyTargetActions(
  rows: TargetScheduleAction[],
  predictionTime: Date,
): TargetSelection {
  const from = predictionTime.getTime() + 10 * 60 * 1000;
  const to = predictionTime.getTime() + 15 * 60 * 1000;
  const inHorizon = rows.filter((row) => {
    const time = new Date(row.target_time_begin).getTime();
    return time > from && time <= to;
  });
  if (inHorizon.length === 0) {
    return { status: 'none', action: null, candidates: 0 };
  }
  const earliest = Math.min(...inHorizon.map((row) => new Date(row.target_time_begin).getTime()));
  const atEarliest = inHorizon
    .filter((row) => new Date(row.target_time_begin).getTime() === earliest)
    .sort((a, b) => numericActionId(a) - numericActionId(b));
  const first = atEarliest[0];
  const equivalent = atEarliest.every(
    (row) => row.target_lat === first.target_lat && row.target_lon === first.target_lon && row.manual_fill === first.manual_fill,
  );
  if (!equivalent) {
    return { status: 'ambiguous', action: null, candidates: atEarliest.length };
  }
  return { status: 'ok', action: first, candidates: atEarliest.length };
}

/** Backwards-compatible helper: the unique (or proven-equivalent) target, else null. */
export function resolveTargetActionFromRows(
  rows: TargetScheduleAction[],
  predictionTime: Date,
): TargetScheduleAction | null {
  return classifyTargetActions(rows, predictionTime).action;
}

export type CurrentDeviationStatus = 'confirmed_fact' | 'no_fact_before_t' | 'unavailable_no_fact_source';

export interface CurrentDeviation {
  /** Contract v1 scalar (P1 semantics; 0 when no fact exists or no source is available). */
  seconds: number;
  /** Distinguishes a known value (incl. a known "no fact yet" 0) from an unavailable input. */
  status: CurrentDeviationStatus;
}

export const DEGRADED_CURRENT_DEVIATION_REASON = 'degraded:current_deviation_unavailable(no_fact_source)';

/**
 * Where factual passage times come from. No runtime component writes
 * schedule_actions.time_fact_begin (the DB seed inserts NULL), so the default is 'none':
 * the scalar sent is the Contract-compatible 0, flagged UNAVAILABLE rather than presented
 * as a confirmed "on time". 'replay_import' = organizer TRAIN-day facts were explicitly
 * imported for a historical replay (scripts/import_db_seed.py --replay-facts); it is never
 * a live fact source.
 */
export type ScheduleFactSource = 'none' | 'replay_import';

export function scheduleFactSourceFromEnv(env: Record<string, string | undefined> = process.env): ScheduleFactSource {
  const value = (env.SCHEDULE_FACT_SOURCE ?? 'none').trim().toLowerCase();
  if (value !== 'none' && value !== 'replay_import') {
    throw new Error(`Unsupported SCHEDULE_FACT_SOURCE=${value}; expected "none" or "replay_import"`);
  }
  return value;
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
  ): Promise<TargetSelection> {
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

    return classifyTargetActions(rows, predictionTime);
  }

  async getCurrentDeviation(
    trId: string,
    predictionTime: Date,
    factSource: ScheduleFactSource = scheduleFactSourceFromEnv(),
  ): Promise<CurrentDeviation> {
    if (factSource === 'none') {
      return { seconds: 0, status: 'unavailable_no_fact_source' };
    }
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

    const hasFact = rows.some((row) => row.time_fact_begin && new Date(row.time_fact_begin) <= predictionTime);
    return {
      seconds: resolveCurrentDeviationFromRows(rows, predictionTime),
      status: hasFact ? 'confirmed_fact' : 'no_fact_before_t',
    };
  }
}
