import { Injectable } from '@nestjs/common';
import { Prisma } from '../generated/prisma/client.js';
import { PrismaService } from '../prisma/prisma.service.js';

export interface TripMatch {
  trId: string;
  targetActionId: string;
  distanceMeters: number;
}

@Injectable()
export class TripMatcherService {
  constructor(
    private readonly prisma: PrismaService,
  ) {}

  async findTrip(
    latitude: number,
    longitude: number,
    unitId: string | undefined,
    predictionTime: Date,
  ): Promise<TripMatch | null> {
    if (!(predictionTime instanceof Date) || Number.isNaN(predictionTime.getTime())) {
      throw new Error('findTrip requires an explicit valid prediction time');
    }

    const preferredVehicle = unitId
      ? await this.prisma.vehicles.findUnique({
          where: { unit_id: unitId },
        })
      : null;

    // A trusted vehicles.current_tr_id is the authoritative trip identity.
    // For a trusted assignment we do NOT require the live GPS point to be
    // within 1 km of the schedule geometry: the organizer emulator generates
    // synthetic/random coordinates that are not route-matched.
    // The strict 10–15 minute target horizon remains mandatory.
    const preferredTrId = preferredVehicle?.current_tr_id ?? null;
    const from = new Date(predictionTime.getTime() + 10 * 60 * 1000);
    const to = new Date(predictionTime.getTime() + 15 * 60 * 1000);

    const rows = await this.prisma.$queryRaw<TripMatch[]>`
      WITH valid_trips AS (
        SELECT DISTINCT tr_id
        FROM schedule_actions
        WHERE tr_id NOT LIKE '9000%'
          AND time_begin > ${from}
          AND time_begin <= ${to}
      )
      SELECT
        sa.tr_id AS "trId",
        sa.tt_action_item_id AS "targetActionId",
        ST_Distance(
          sa.geom::geography,
          ST_SetSRID(
            ST_MakePoint(${longitude}, ${latitude}),
            4326
          )::geography
        ) AS "distanceMeters"
      FROM schedule_actions sa
      JOIN valid_trips vt ON vt.tr_id = sa.tr_id
      WHERE ${preferredTrId ? Prisma.sql`sa.tr_id = ${preferredTrId}` : Prisma.sql`TRUE`}
        AND ${preferredTrId
          ? Prisma.sql`TRUE`
          : Prisma.sql`ST_DWithin(
              sa.geom::geography,
              ST_SetSRID(
                ST_MakePoint(${longitude}, ${latitude}),
                4326
              )::geography,
              1000
            )`}
      ORDER BY
        sa.geom::geography <-> ST_SetSRID(
          ST_MakePoint(${longitude}, ${latitude}),
          4326
        )::geography,
        sa.time_begin ASC
      LIMIT 1
    `;

    if (rows.length > 0) {
      return rows[0];
    }

    // When there is no trusted trip assignment, keep the original spatial
    // matching behavior. Do not invent a route for an unresolvable vehicle.
    if (preferredTrId) {
      return null;
    }

    const rowsFallback = await this.prisma.$queryRaw<TripMatch[]>`
      WITH valid_trips AS (
        SELECT DISTINCT tr_id
        FROM schedule_actions
        WHERE tr_id NOT LIKE '9000%'
          AND time_begin > ${from}
          AND time_begin <= ${to}
      )
      SELECT
        sa.tr_id AS "trId",
        sa.tt_action_item_id AS "targetActionId",
        ST_Distance(
          sa.geom::geography,
          ST_SetSRID(
            ST_MakePoint(${longitude}, ${latitude}),
            4326
          )::geography
        ) AS "distanceMeters"
      FROM schedule_actions sa
      JOIN valid_trips vt ON vt.tr_id = sa.tr_id
      WHERE ST_DWithin(
          sa.geom::geography,
          ST_SetSRID(
            ST_MakePoint(${longitude}, ${latitude}),
            4326
          )::geography,
          1000
        )
      ORDER BY
        sa.geom::geography <-> ST_SetSRID(
          ST_MakePoint(${longitude}, ${latitude}),
          4326
        )::geography,
        sa.time_begin ASC
      LIMIT 1
    `;

    return rowsFallback[0] ?? null;
  }
}
