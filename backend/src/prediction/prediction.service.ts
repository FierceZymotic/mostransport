import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service.js';
import { TelemetryRepository } from '../telemetry/telemetry.repository.js';
import { MlClientService } from './ml/ml.client.service.js';
import {
  PredictionRequest,
  PredictionResponse,
} from './ml/ml.types.js';
import { ScheduleRepository } from './schedule.repository.js';
import { TripMatcherService } from './trip-matcher.service.js';

export const DEFAULT_DEMO_PREDICTION_TIME = new Date('2026-01-06T03:35:00.000Z');

export function normalizeTimestampToFrozenContract(
  value: Date | number,
): number {
  const timestamp = value instanceof Date ? value.getTime() : Number(value);

  if (!Number.isFinite(timestamp)) {
    return Math.floor(DEFAULT_DEMO_PREDICTION_TIME.getTime() / 1000);
  }

  const candidate = new Date(timestamp);

  const frozenStart = new Date('2026-01-06T00:00:00.000Z');
  const frozenEnd = new Date('2026-01-07T00:00:00.000Z');

  // Исторические данные 6 января оставляем как есть.
  if (candidate >= frozenStart && candidate < frozenEnd) {
    return Math.floor(candidate.getTime() / 1000);
  }

  // Live/demo: переносим текущую дату на 6 января,
  // сохраняя время суток.
  const demoTime = new Date(
    Date.UTC(
      2026,
      0,
      6,
      candidate.getUTCHours(),
      candidate.getUTCMinutes(),
      candidate.getUTCSeconds(),
      candidate.getUTCMilliseconds(),
    ),
  );

  return Math.floor(demoTime.getTime() / 1000);
}

export function normalizePredictionTime(input: Date): Date {
  const value = new Date(input);

  if (Number.isNaN(value.getTime())) {
    return new Date(DEFAULT_DEMO_PREDICTION_TIME);
  }

  const frozenStart = new Date('2026-01-06T00:00:00.000Z');
  const frozenEnd = new Date('2026-01-07T00:00:00.000Z');
  if (value >= frozenStart && value < frozenEnd) {
    return value;
  }

  return new Date(DEFAULT_DEMO_PREDICTION_TIME);
}

export function normalizeTelemetryTimestamp(input: number | Date): number {
  return normalizeTimestampToFrozenContract(input);
}

@Injectable()
export class PredictionService {
  constructor(
    private readonly mlClient: MlClientService,
    private readonly telemetryRepository: TelemetryRepository,
    private readonly prisma: PrismaService,
    private readonly scheduleRepository: ScheduleRepository,
    private readonly tripMatcher: TripMatcherService,
  ) {}

  async predictForVehicle(
    unitId: number,
  ): Promise<PredictionResponse> {
    return this.predictForVehicleAt(
      unitId,
      DEFAULT_DEMO_PREDICTION_TIME,
    );
  }

  async predictForVehicleAt(
    unitId: number,
    predictionTime: Date,
  ): Promise<PredictionResponse> {
    const normalizedPredictionTime = normalizePredictionTime(predictionTime);

    const vehicleHistory =
      await this.telemetryRepository.findHistory(
        unitId,
        normalizedPredictionTime,
      );

    if (vehicleHistory.length === 0) {
      throw new Error(
        `No telemetry found for unit ${unitId} at ${normalizedPredictionTime.toISOString()}`,
      );
    }

    const latest = vehicleHistory[vehicleHistory.length - 1];

    const match = await this.tripMatcher.findTrip(
  latest.latitude,
  latest.longitude,
);
    if (!match) {
      throw new Error(
        `Could not determine trip for unit ${unitId}`,
      );
    }

    const targetAction =
      await this.scheduleRepository.findTargetAction(
        match.trId,
        normalizedPredictionTime,
      );

    if (!targetAction) {
      throw new Error(
        `No target schedule action found for trId ${match.trId} at ${normalizedPredictionTime.toISOString()}`,
      );
    }

    const request: PredictionRequest = {
      request_id: crypto.randomUUID(),

      prediction_time: normalizedPredictionTime.toISOString(),

      vehicle_context: {
        unit_id: String(unitId),
        tr_id: match.trId,
        route_id: match.trId,
      },

      schedule_context: {
        target_action_id:
          targetAction.target_action_id,

        target_time_begin: new Date(
          targetAction.target_time_begin,
        ).toISOString(),

        target_lat: targetAction.target_lat,
        target_lon: targetAction.target_lon,

        current_deviation_seconds:
  await this.scheduleRepository.getCurrentDeviation(
    match.trId,
    normalizedPredictionTime,
  ),

        manual_fill: targetAction.manual_fill,
      },

      telemetry: vehicleHistory
        .filter(
          (state) =>
            state.timestamp <=
            Math.floor(normalizedPredictionTime.getTime() / 1000),
        )
        .map((state) => ({
          event_time: new Date(
            state.timestamp * 1000,
          ).toISOString(),

          lat: state.locationValid
            ? state.latitude
            : null,

          lon: state.locationValid
            ? state.longitude
            : null,

          location_valid: state.locationValid,

          speed: state.speed,
        })),
    };

    const response = await this.mlClient.predict(request);

await this.prisma.predictions.create({
  data: {
    request_id: response.request_id,
    unit_id: String(unitId),
    tr_id: match.trId,
    prediction_time: normalizedPredictionTime,
    target_action_id: targetAction.target_action_id,
    target_time_begin: new Date(targetAction.target_time_begin),
    status: response.status,
    delay_seconds: response.prediction.delay_seconds,
    target_time: new Date(response.prediction.target_time),
    reason: response.prediction.reason,
    generated_at: new Date(response.generated_at),
    model_version: response.model_version,
    feature_schema_version: response.feature_schema_version,
  },
});

return response;

  }

  async testDatabase(): Promise<number> {
    return this.prisma.vehicles.count();
  }
}

