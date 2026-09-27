import { Injectable } from '@nestjs/common';
import { normalizeTelemetryTimestamp } from '../prediction/prediction.service.js';
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

  async findHistory(
    unitId: number,
    predictionTime: Date,
  ): Promise<VehicleState[]> {
    const from = new Date(predictionTime.getTime() - 15 * 60 * 1000);

    const rows = await this.prisma.telemetry.findMany({
      where: {
        unit_id: String(unitId),
        timestamp: {
          lte: predictionTime,
        },
      },
      orderBy: {
        timestamp: 'asc',
      },
    });

    const recentRows = rows.filter((row) => row.timestamp >= from);
    const lastPacket = rows.at(-1);
    const lastValidGps = [...rows].reverse().find(
      (row) => row.location_valid && row.longitude != null && row.latitude != null,
    );

    const deduped = new Map<number, (typeof rows)[number]>();

    for (const row of [...recentRows, ...(lastPacket ? [lastPacket] : []), ...(lastValidGps ? [lastValidGps] : [])]) {
      deduped.set(row.timestamp.getTime(), row);
    }

    return [...deduped.values()]
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
      .map((row) => ({
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