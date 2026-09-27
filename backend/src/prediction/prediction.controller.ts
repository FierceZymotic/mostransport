import { Controller, Get, Param, Query } from '@nestjs/common';
import { demoClock } from '../common/demo-clock.js';
import { PrismaService } from '../prisma/prisma.service.js';
import { PredictionService } from './prediction.service.js';
import { ScheduleRepository } from './schedule.repository.js';
import { TripMatcherService } from './trip-matcher.service.js';

@Controller('prediction')
export class PredictionController {
  constructor(
  private readonly predictionService: PredictionService,
  private readonly scheduleRepository: ScheduleRepository,
  private readonly tripMatcherService: TripMatcherService,
  private readonly prisma: PrismaService,
) {}

  /**
   * Real trip id of a unit at its last packet time T via the canonical live resolution (the same
   * one live prediction uses), independent of ML and prediction history; null when the trip
   * cannot be determined. Never a placeholder.
   */
  private resolveRoute(unitId: string, lastPacketTime: Date): Promise<string | null> {
    return this.safeRead(
      async () => (await this.predictionService.resolveTrip(Number(unitId), lastPacketTime)).match?.trId ?? null,
      null,
    );
  }

  private async safeRead<T>(operation: () => Promise<T>, fallback: T): Promise<T> {
    try {
      return await operation();
    } catch (error) {
      console.warn('Dashboard data read failed; returning fallback payload.', error);
      return fallback;
    }
  }

@Get('run/:unitId')
async run(@Param('unitId') unitId: string) {
  return this.predictionService.predictForVehicle(
    Number(unitId),
  );
}

  @Get('db-test')
  async dbTest() {
    return this.predictionService.testDatabase();
  }

  @Get('schedule-test')
  async scheduleTest(
    @Query('trId') trId: string,
    @Query('time') time: string,
  ) {
    try {
      return await this.scheduleRepository.findTargetAction(
        trId,
        new Date(time),
      );
    } catch (error) {
      console.error('SCHEDULE TEST ERROR:', error);
      throw error;
    }
  }
  @Get('matcher-test')
async matcherTest(
  @Query('lat') lat: string,
  @Query('lon') lon: string,
  @Query('unitId') unitId?: string,
  @Query('time') time?: string,
) {
  return this.tripMatcherService.findTrip(
    Number(lat),
    Number(lon),
    unitId,
    time ? new Date(time) : demoClock.now(),
  );
}
@Get('run-at/:unitId')
async runAt(
  @Param('unitId') unitId: string,
  @Query('time') time: string,
) {
  return this.predictionService.predictForVehicleAt(
    Number(unitId),
    new Date(time),
  );
}

@Get('dashboard/summary')
async dashboardSummary() {
  const activeVehicles = await this.safeRead(
    () => this.prisma.vehicle_last_state.findMany({
      take: 50,
      orderBy: { timestamp: 'desc' },
    }),
    [],
  );

  const vehicles = await Promise.all(activeVehicles.map(async (row) => ({
    id: String(row.unit_id),
    route: await this.resolveRoute(row.unit_id, row.timestamp),
    lat: row.latitude ?? 0,
    lon: row.longitude ?? 0,
    speed: row.speed ?? 0,
    delayMinutes: row.speed != null && row.speed < 10 ? 5 : 0,
    risk: row.speed != null && row.speed < 10 ? 'high' : row.speed != null && row.speed < 20 ? 'medium' : 'low',
    reason: row.speed != null && row.speed < 10
      ? 'Низкая скорость в текущем состоянии'
      : row.speed != null && row.speed < 20
        ? 'Некоторое отклонение от графика'
        : 'Отклонений не обнаружено',
    segment: row.location_valid ? 'Realtime telemetry' : 'Awaiting GPS',
    updatedAt: row.timestamp.toISOString(),
  })));

  const currentRisk = vehicles.filter((vehicle) => vehicle.risk === 'high').length;

  return {
    total: vehicles.length,
    highRisk: currentRisk,
    mediumRisk: vehicles.filter((vehicle) => vehicle.risk === 'medium').length,
    lowRisk: vehicles.filter((vehicle) => vehicle.risk === 'low').length,
    vehicles,
  };
}

@Get('dashboard/vehicles')
async dashboardVehicles() {
  const rows = await this.safeRead(
    () => this.prisma.vehicle_last_state.findMany({
      take: 50,
      orderBy: { timestamp: 'desc' },
    }),
    [],
  );

  return Promise.all(rows.map(async (row) => ({
    id: String(row.unit_id),
    route: await this.resolveRoute(row.unit_id, row.timestamp),
    lat: row.latitude ?? 0,
    lon: row.longitude ?? 0,
    speed: row.speed ?? 0,
    locationValid: row.location_valid ?? false,
    updatedAt: row.timestamp.toISOString(),
  })));
}

@Get('dashboard/alerts')
async dashboardAlerts() {
  const rows = await this.safeRead(
    () => this.prisma.predictions.findMany({
      take: 10,
      orderBy: { generated_at: 'desc' },
    }),
    [],
  );

  return rows.map((row) => ({
    id: row.request_id,
    unitId: String(row.unit_id),
    routeId: row.tr_id ?? 'unknown',
    status: row.status,
    delaySeconds: row.delay_seconds ?? 0,
    reason: row.reason ?? 'prediction',
    generatedAt: row.generated_at.toISOString(),
  }));
}
}