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
    unitId?: string,
  ): Promise<TripMatch | null> {
    const preferredVehicle = unitId
      ? await this.prisma.vehicles.findUnique({
          where: { unit_id: unitId },
        })
      : null;

    const preferredTrId = preferredVehicle?.current_tr_id ?? null;

    const rows = await this.prisma.$queryRaw<TripMatch[]>`
      SELECT
        tr_id AS "trId",
        tt_action_item_id AS "targetActionId",
        ST_Distance(
          geom::geography,
          ST_SetSRID(
            ST_MakePoint(${longitude}, ${latitude}),
            4326
          )::geography
        ) AS "distanceMeters"
      FROM schedule_actions
      WHERE ${preferredTrId ? Prisma.sql`tr_id = ${preferredTrId}` : Prisma.sql`TRUE`}
        AND ST_DWithin(
          geom::geography,
          ST_SetSRID(
            ST_MakePoint(${longitude}, ${latitude}),
            4326
          )::geography,
          1000
        )
      ORDER BY
        geom::geography <-> ST_SetSRID(
          ST_MakePoint(${longitude}, ${latitude}),
          4326
        )::geography,
        CAST(tt_action_item_id AS bigint) ASC
      LIMIT 1
    `;

    if (rows.length > 0) {
      return rows[0];
    }

    if (preferredTrId) {
      const rowsFallback = await this.prisma.$queryRaw<TripMatch[]>`
        SELECT
          tr_id AS "trId",
          tt_action_item_id AS "targetActionId",
          ST_Distance(
            geom::geography,
            ST_SetSRID(
              ST_MakePoint(${longitude}, ${latitude}),
              4326
            )::geography
          ) AS "distanceMeters"
        FROM schedule_actions
        WHERE ST_DWithin(
          geom::geography,
          ST_SetSRID(
            ST_MakePoint(${longitude}, ${latitude}),
            4326
          )::geography,
          1000
        )
        ORDER BY
          geom::geography <-> ST_SetSRID(
            ST_MakePoint(${longitude}, ${latitude}),
            4326
          )::geography,
          CAST(tt_action_item_id AS bigint) ASC
        LIMIT 1
      `;

      return rowsFallback[0] ?? null;
    }

    return null;
  }
}