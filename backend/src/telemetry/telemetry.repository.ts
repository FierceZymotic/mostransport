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
}