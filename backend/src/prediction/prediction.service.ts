import { Injectable } from '@nestjs/common';
import { TelemetryHistory } from '../telemetry/telemetry-history.js';
import { MlClientService } from './ml/ml.client.service.js';
import {
  PredictionRequest,
  PredictionResponse,
} from './ml/ml.types.js';
import { PrismaService } from '../prisma/prisma.service.js';

@Injectable()
export class PredictionService {
  constructor(
  private readonly mlClient: MlClientService,
  private readonly telemetryHistory: TelemetryHistory,
  private readonly prisma: PrismaService,
) {}

  
  async predictForVehicle(
    unitId: number,
  ): Promise<PredictionResponse> {
    const history = this.telemetryHistory.getAll();

    const vehicleHistory = history.filter(
      (state) => state.unitId === unitId,
    );

    if (vehicleHistory.length === 0) {
      throw new Error(
        `No telemetry found for unit ${unitId}`,
      );
    }

    const latest = vehicleHistory[vehicleHistory.length - 1];

    const request: PredictionRequest = {
      request_id: crypto.randomUUID(),
      prediction_time: new Date(
        latest.timestamp * 1000,
      ).toISOString(),

      vehicle_context: {
        unit_id: String(unitId),
        tr_id: 'unknown',
        route_id: 'unknown',
      },

      schedule_context: {
        target_action_id: 'unknown',
        target_time_begin: new Date(
          latest.timestamp * 1000 + 10 * 60 * 1000,
        ).toISOString(),
        target_lat: latest.latitude,
        target_lon: latest.longitude,
        current_deviation_seconds: 0,
        manual_fill: false,
      },

      telemetry: this.telemetryHistory
        .getTelemetryForPrediction(
          unitId,
          latest.timestamp,
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
     

    return this.mlClient.predict(request);
  }
  async testDatabase(): Promise<number> {
  return this.prisma.vehicles.count();
}
}