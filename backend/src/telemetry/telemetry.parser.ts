import { parseNav00 } from './ndtp/cells/nav00.js';
import { VehicleState } from './vehicle-state.js';

/**
 * A validity flag alone does not make a position: the emulator inserts synthetic navigation
 * ("zeros, flags N/E/valid", organizer spec §3.2) when no G6CellNav00 is configured, and a
 * raw u32 magnitude / 1e7 can exceed the coordinate range. Such cells stay packets but are
 * not strict-valid GPS.
 */
export function isPlausiblePosition(latitude: number, longitude: number): boolean {
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return false;
  if (Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return false;
  return !(latitude === 0 && longitude === 0);
}

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
      locationValid: nav.locationValid && isPlausiblePosition(nav.latitude, nav.longitude),

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