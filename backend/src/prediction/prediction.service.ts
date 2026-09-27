import { BadRequestException, Injectable, UnprocessableEntityException } from '@nestjs/common';
import { demoClock } from '../common/demo-clock.js';
import { PrismaService } from '../prisma/prisma.service.js';
import { TelemetryRepository } from '../telemetry/telemetry.repository.js';
import { MlClientService } from './ml/ml.client.service.js';
import {
  PredictionRequest,
  PredictionResponse,
} from './ml/ml.types.js';
import {
  CurrentDeviationStatus,
  DEGRADED_CURRENT_DEVIATION_REASON,
  ScheduleRepository,
} from './schedule.repository.js';
import { TripMatch, TripMatcherService } from './trip-matcher.service.js';
import { VehicleState } from '../telemetry/vehicle-state.js';

export class InvalidPredictionTimeError extends BadRequestException {}

export type PredictionIneligibleCode =
  | 'NO_TELEMETRY'
  | 'NO_VALID_GPS'
  | 'NO_TRIP_MATCH'
  | 'NO_TARGET_IN_HORIZON'
  | 'TARGET_AMBIGUOUS';

/** The point is not eligible for a Contract v1 request (explicit 422, not a crash or a guess). */
export class PredictionIneligibleError extends UnprocessableEntityException {
  constructor(readonly code: PredictionIneligibleCode, detail: string) {
    super({ code, detail });
  }
}

/**
 * Validates an explicit model-time prediction instant T. T is never substituted: the
 * Backend clock (common/demo-clock.ts) is the only place where live time is mapped.
 */
export function requirePredictionTime(input: Date): Date {
  const value = new Date(input);
  if (Number.isNaN(value.getTime())) {
    throw new InvalidPredictionTimeError('prediction time must be a valid ISO-8601 timestamp');
  }
  return value;
}

/** Outcome of the canonical live trip resolution (see PredictionService.resolveTrip). */
export type TripResolution =
  | { status: 'resolved'; match: TripMatch }
  | { status: 'NO_VALID_GPS' | 'NO_TRIP_MATCH'; match: null };

/** ML response plus Backend-side input provenance that Contract v1 cannot carry. */
export interface PredictionOutcome {
  response: PredictionResponse;
  currentDeviationStatus: CurrentDeviationStatus;
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
    // "Now" in model time: the same session clock that stamps ingested telemetry.
    return this.predictForVehicleAt(
      unitId,
      demoClock.now(),
    );
  }

  /**
   * Canonical live trip identity of a unit at model time T, shared by live prediction and the
   * dashboard: the latest strict-valid GPS at or before T, matched by TripMatcherService (mapped
   * vehicles.current_tr_id first; only trips with an action in (T+10m, T+15m]; no synthetic
   * 9000xxx trips). Independent of ML and of prediction history. `history` (the Contract history
   * at T, which always keeps that GPS anchor) avoids a second query when the caller has it.
   */
  async resolveTrip(
    unitId: number,
    predictionTime: Date,
    history?: VehicleState[],
  ): Promise<TripResolution> {
    const position = history
      ? [...history].reverse().find(
          (state) => state.locationValid && Number.isFinite(state.latitude) && Number.isFinite(state.longitude),
        ) ?? null
      : await this.telemetryRepository.findLatestValidPosition(unitId, predictionTime);
    if (!position) {
      return { status: 'NO_VALID_GPS', match: null };
    }
    const match = await this.tripMatcher.findTrip(
      position.latitude,
      position.longitude,
      String(unitId),
      predictionTime,
    );
    return match ? { status: 'resolved', match } : { status: 'NO_TRIP_MATCH', match: null };
  }

  async predictForVehicleAt(
    unitId: number,
    predictionTime: Date,
  ): Promise<PredictionResponse> {
    return (await this.predictWithStatus(unitId, predictionTime)).response;
  }

  async predictWithStatus(
    unitId: number,
    predictionTime: Date,
  ): Promise<PredictionOutcome> {
    const normalizedPredictionTime = requirePredictionTime(predictionTime);

    const vehicleHistory =
      await this.telemetryRepository.findHistory(
        unitId,
        normalizedPredictionTime,
      );

    if (vehicleHistory.length === 0) {
      throw new PredictionIneligibleError(
        'NO_TELEMETRY',
        `No telemetry found for unit ${unitId} at ${normalizedPredictionTime.toISOString()}`,
      );
    }

    // Same canonical trip identity the dashboard shows (latest strict-valid GPS <= T; an
    // invalid fix carries no trustworthy location).
    const trip = await this.resolveTrip(unitId, normalizedPredictionTime, vehicleHistory);
    if (trip.status !== 'resolved') {
      throw new PredictionIneligibleError(
        trip.status,
        trip.status === 'NO_VALID_GPS'
          ? `No strict-valid GPS <= T for unit ${unitId}`
          : `Could not determine trip for unit ${unitId}`,
      );
    }
    const match = trip.match;

    const selection =
      await this.scheduleRepository.findTargetAction(
        match.trId,
        normalizedPredictionTime,
      );

    if (selection.status === 'ambiguous') {
      throw new PredictionIneligibleError(
        'TARGET_AMBIGUOUS',
        `${selection.candidates} different planned actions share the earliest time in (T+10m, T+15m] for trId ${match.trId}`,
      );
    }
    const targetAction = selection.action;
    if (!targetAction) {
      throw new PredictionIneligibleError(
        'NO_TARGET_IN_HORIZON',
        `No target schedule action found for trId ${match.trId} at ${normalizedPredictionTime.toISOString()}`,
      );
    }

    // P1 semantics. Without a fact source the Contract v1 scalar is 0 and the status says
    // "unavailable"; it is never presented as a confirmed on-time value.
    const deviation = await this.scheduleRepository.getCurrentDeviation(
      match.trId,
      normalizedPredictionTime,
    );

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

        current_deviation_seconds: deviation.seconds,

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
    // Degraded-mode marker (additive; Contract v1 unchanged): the model input
    // current_deviation_seconds was not backed by any fact source.
    reason: response.prediction.reason
      ?? (deviation.status === 'unavailable_no_fact_source' ? DEGRADED_CURRENT_DEVIATION_REASON : null),
    generated_at: new Date(response.generated_at),
    model_version: response.model_version,
    feature_schema_version: response.feature_schema_version,
  },
});

return { response, currentDeviationStatus: deviation.status };

  }

  async testDatabase(): Promise<number> {
    return this.prisma.vehicles.count();
  }
}

