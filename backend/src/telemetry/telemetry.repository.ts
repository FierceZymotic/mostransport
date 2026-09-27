import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service.js';
import { VehicleState } from './vehicle-state.js';

@Injectable()
export class TelemetryRepository {
  constructor(private readonly prisma: PrismaService) {}

  async save(state: VehicleState): Promise<void> {
    
    await this.prisma.telemetry.create({
      data: {
        unit_id: String(state.unitId),
        timestamp: new Date(state.timestamp * 1000),
        longitude: state.longitude,
        latitude: state.latitude,
        location_valid: state.locationValid,
        speed: state.speed,
        speed_max: state.speedMax,
        course: state.course,
        track: state.track,
        altitude: state.altitude,
        nsat: state.nsat,
        pdop: state.pdop,
      },
    });
  }

  async saveLastState(state: VehicleState): Promise<void> {
  await this.prisma.vehicle_last_state.upsert({
    where: {
      unit_id: String(state.unitId),
    },
    create: {
      unit_id: String(state.unitId),
      tr_id: null,
      timestamp: new Date(state.timestamp * 1000),
      longitude: state.longitude,
      latitude: state.latitude,
      speed: state.speed,
      location_valid: state.locationValid,
    },
    update: {
      timestamp: new Date(state.timestamp * 1000),
      longitude: state.longitude,
      latitude: state.latitude,
      speed: state.speed,
      location_valid: state.locationValid,
    },
  });
}

  async findLatestTimestamp(
    unitId: number,
  ): Promise<Date | null> {
    const row = await this.prisma.telemetry.findFirst({
      where: {
        unit_id: String(unitId),
      },
      orderBy: {
        timestamp: 'desc',
      },
      select: {
        timestamp: true,
      },
    });

    return row?.timestamp ?? null;
  }

  /**
   * Latest strict-valid GPS fix (location_valid and finite coordinates) at or before T: the same
   * anchor selectContractHistory always keeps, read without loading the whole history.
   */
  async findLatestValidPosition(
    unitId: number,
    predictionTime: Date,
  ): Promise<{ latitude: number; longitude: number } | null> {
    const pageSize = 50;
    for (let skip = 0; ; skip += pageSize) {
      const rows = await this.prisma.telemetry.findMany({
        where: {
          unit_id: String(unitId),
          timestamp: { lte: predictionTime },
          location_valid: true,
        },
        orderBy: [{ timestamp: 'desc' }, { id: 'desc' }],
        skip,
        take: pageSize,
        select: { latitude: true, longitude: true },
      });
      const fix = rows.find((row) => Number.isFinite(row.latitude) && Number.isFinite(row.longitude));
      if (fix) return fix;
      if (rows.length < pageSize) return null;
    }
  }

  async findHistory(
    unitId: number,
    predictionTime: Date,
  ): Promise<VehicleState[]> {
    const rows = await this.prisma.telemetry.findMany({
      where: {
        unit_id: String(unitId),
        timestamp: {
          lte: predictionTime,
        },
      },
      // (timestamp, id): id is the insertion (arrival) order, a deterministic tie-break.
      orderBy: [{ timestamp: 'asc' }, { id: 'asc' }],
    });

    return selectContractHistory(rows, predictionTime).map((row) => ({
      unitId: Number(row.unit_id),
      timestamp: Math.floor(row.timestamp.getTime() / 1000),
      longitude: row.longitude,
      latitude: row.latitude,
      locationValid: row.location_valid,
      speed: row.speed ?? 0,
      speedMax: row.speed_max ?? 0,
      course: row.course ?? 0,
      track: row.track ?? 0,
      altitude: row.altitude ?? 0,
      nsat: row.nsat ?? 0,
      pdop: row.pdop ?? 0,
    }));
  }
}

export interface HistoryRow {
  id: bigint | number;
  timestamp: Date;
  location_valid: boolean;
  longitude: number | null;
  latitude: number | null;
}

const FIFTEEN_MINUTES_MS = 15 * 60 * 1000;

function compareRows(a: HistoryRow, b: HistoryRow): number {
  const dt = a.timestamp.getTime() - b.timestamp.getTime();
  if (dt !== 0) {
    return dt;
  }
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/**
 * Contract v1 request history (BACKEND_ML_INTEGRATION §7): every packet with
 * event_time in (T-15m, T], plus the last packet <= T and the last strict-valid GPS
 * packet <= T. Packets after T never enter. Distinct packets that share a timestamp are
 * all kept (deduplication is by row identity, not by timestamp). Order is (timestamp, id).
 */
export function selectContractHistory<R extends HistoryRow>(rows: R[], predictionTime: Date): R[] {
  const cutoff = predictionTime.getTime();
  const windowStart = cutoff - FIFTEEN_MINUTES_MS;
  const past = rows.filter((row) => row.timestamp.getTime() <= cutoff).sort(compareRows);
  if (past.length === 0) {
    return [];
  }
  const keep = new Set<R>(past.filter((row) => row.timestamp.getTime() > windowStart));
  keep.add(past[past.length - 1]);
  for (let i = past.length - 1; i >= 0; i--) {
    const row = past[i];
    if (row.location_valid && Number.isFinite(row.longitude) && Number.isFinite(row.latitude)) {
      keep.add(row);
      break;
    }
  }
  return past.filter((row) => keep.has(row));
}
