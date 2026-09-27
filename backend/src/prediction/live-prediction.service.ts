import {
  Injectable,
  Logger,
  OnModuleDestroy,
  OnModuleInit,
} from '@nestjs/common';
import { TelemetryStreamService } from '../telemetry/telemetry-stream.service.js';
import { VehicleState } from '../telemetry/vehicle-state.js';
import { PredictionService } from './prediction.service.js';

@Injectable()
export class LivePredictionService
  implements OnModuleInit, OnModuleDestroy
{
  private readonly logger = new Logger(LivePredictionService.name);

  private readonly predictionIntervalSeconds = 60;

  private readonly lastPredictionAt = new Map<number, number>();

  private readonly inFlight = new Set<number>();

  private unsubscribe?: () => void;

  constructor(
    private readonly stream: TelemetryStreamService,
    private readonly predictionService: PredictionService,
  ) {}

  onModuleInit() {
    this.unsubscribe = this.stream.subscribeState(
      (state) => {
        void this.handleTelemetry(state);
      },
    );

    this.logger.log('Live prediction service started');
  }

  onModuleDestroy() {
    this.unsubscribe?.();
  }

  private async handleTelemetry(state: VehicleState): Promise<void> {
    // Для prediction нам нужны валидные координаты.
    if (!state.locationValid) {
      return;
    }

    const now = state.timestamp;

    const lastPrediction = this.lastPredictionAt.get(state.unitId);

    if (
      lastPrediction !== undefined &&
      now - lastPrediction < this.predictionIntervalSeconds
    ) {
      return;
    }

    // Не запускаем второй prediction для того же ТС,
    // пока предыдущий ещё выполняется.
    if (this.inFlight.has(state.unitId)) {
      return;
    }

    this.inFlight.add(state.unitId);

    try {
      const prediction =
        await this.predictionService.predictForVehicleAt(
          state.unitId,
          new Date(state.timestamp * 1000),
        );

      this.lastPredictionAt.set(state.unitId, now);

      this.stream.publishPrediction(prediction);

      this.logger.log(
        `Live prediction: unit=${state.unitId}, delay=${prediction.prediction.delay_seconds}s`,
      );
    } catch (error) {
      this.logger.debug(
        `Live prediction skipped for unit ${state.unitId}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    } finally {
      this.inFlight.delete(state.unitId);
    }
  }
}