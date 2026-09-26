import { parseNav00 } from './ndtp/cells/nav00.js';
import { VehicleState } from './vehicle-state.js';

export class TelemetryParser {
  parse(
    payload: Buffer,
    unitId: number,
  ): VehicleState {

    console.log(
      'TelemetryParser payload length:',
      payload.length,
    );

    console.log(
      'TelemetryParser payload hex:',
      payload.toString('hex'),
    );

    const nav = parseNav00(payload);

    return {
      unitId,

      timestamp: nav.timestamp,

      longitude: nav.longitude,
      latitude: nav.latitude,
      locationValid: nav.locationValid,

      speed: nav.speedAvg,
      speedMax: nav.speedMax,

      course: nav.course,
      track: nav.track,

      altitude: nav.altitude,
      nsat: nav.nsat,
      pdop: nav.pdop,
    };
  }
}