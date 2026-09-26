import { Injectable } from '@nestjs/common';
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
  ): Promise<TripMatch | null> {
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
        )::geography
      LIMIT 1
    `;

    return rows[0] ?? null;
  }
}