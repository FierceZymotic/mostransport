import { Injectable } from '@nestjs/common';
import { VehicleState } from './vehicle-state.js';
import { PredictionResponse } from '../prediction/ml/ml.types.js';

type StreamClient = {
  send: (message: string) => void;
  close: () => void;
};

type StateSubscriber = (state: VehicleState) => void;

@Injectable()
export class TelemetryStreamService {
  private readonly clients = new Set<StreamClient>();

  private readonly stateSubscribers =
    new Set<StateSubscriber>();

  subscribe(client: StreamClient): () => void {
    this.clients.add(client);

    return () => {
      this.clients.delete(client);
    };
  }

  subscribeState(
    subscriber: StateSubscriber,
  ): () => void {
    this.stateSubscribers.add(subscriber);

    return () => {
      this.stateSubscribers.delete(subscriber);
    };
  }

  publish(state: VehicleState): void {
    const payload = JSON.stringify({
      event: 'telemetry',
      payload: {
        unitId: state.unitId,
        timestamp: state.timestamp,
        longitude: state.longitude,
        latitude: state.latitude,
        locationValid: state.locationValid,
        speed: state.speed,
        speedMax: state.speedMax,
        course: state.course,
        track: state.track,
        altitude: state.altitude,
        nsat: state.nsat,
        pdop: state.pdop,
      },
    });

    for (const subscriber of [...this.stateSubscribers]) {
      try {
        subscriber(state);
      } catch {
        this.stateSubscribers.delete(subscriber);
      }
    }

    for (const client of [...this.clients]) {
      try {
        client.send(payload);
      } catch {
        this.clients.delete(client);
      }
    }
  }

  publishPrediction(
    prediction: PredictionResponse,
  ): void {
    const payload = JSON.stringify({
      event: 'prediction',
      payload: prediction,
    });

    for (const client of [...this.clients]) {
      try {
        client.send(payload);
      } catch {
        this.clients.delete(client);
      }
    }
  }
}