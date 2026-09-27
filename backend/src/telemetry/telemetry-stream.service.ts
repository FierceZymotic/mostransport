import { Injectable } from '@nestjs/common';
import { VehicleState } from './vehicle-state.js';

type StreamClient = {
  send: (message: string) => void;
  close: () => void;
};

@Injectable()
export class TelemetryStreamService {
  private readonly clients = new Set<StreamClient>();

  subscribe(client: StreamClient): () => void {
    this.clients.add(client);

    return () => {
      this.clients.delete(client);
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

    for (const client of [...this.clients]) {
      try {
        client.send(payload);
      } catch {
        this.clients.delete(client);
      }
    }
  }
}
