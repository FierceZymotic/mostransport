import { Injectable } from '@nestjs/common';
import { VehicleState } from './vehicle-state.js';

@Injectable()
export class TelemetryHistory {
  private readonly history: VehicleState[] = [];

  add(state: VehicleState): void {
    this.history.push(state);
  }

  getTelemetryForPrediction(
    unitId: number,
    predictionTime: number,
  ): VehicleState[] {
    const fifteenMinutesAgo =
      predictionTime - 15 * 60;

    const vehicleHistory = this.history.filter(
      (state) => state.unitId === unitId,
    );

    const selected = new Set<VehicleState>();

    // Основное окно: (T - 15m, T]
    for (const state of vehicleHistory) {
      if (
        state.timestamp > fifteenMinutesAgo &&
        state.timestamp <= predictionTime
      ) {
        selected.add(state);
      }
    }

    // Последний packet <= T
    const lastPacket = this.findLast(
      vehicleHistory,
      (state) => state.timestamp <= predictionTime,
    );

    // Последний packet со strict-valid GPS <= T
    const lastValidGpsPacket = this.findLast(
      vehicleHistory,
      (state) =>
        state.timestamp <= predictionTime &&
        state.locationValid === true,
    );

    // Последний packet с speed <= T
    const lastSpeedPacket = this.findLast(
      vehicleHistory,
      (state) => state.timestamp <= predictionTime,
    );

    if (lastPacket) {
      selected.add(lastPacket);
    }

    if (lastValidGpsPacket) {
      selected.add(lastValidGpsPacket);
    }

    if (lastSpeedPacket) {
      selected.add(lastSpeedPacket);
    }

    return vehicleHistory.filter(
      (state) => selected.has(state),
    );
  }

  private findLast(
    history: VehicleState[],
    predicate: (state: VehicleState) => boolean,
  ): VehicleState | undefined {
    for (let i = history.length - 1; i >= 0; i--) {
      const state = history[i];

      if (predicate(state)) {
        return state;
      }
    }

    return undefined;
  }

  getAll(): VehicleState[] {
    return [...this.history];
  }
}