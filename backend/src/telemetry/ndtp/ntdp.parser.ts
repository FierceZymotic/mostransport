import { TelemetryEvent } from '../telemetry.types.js';

export interface NdtpParser {
  parse(buffer: Buffer): TelemetryEvent[];
}