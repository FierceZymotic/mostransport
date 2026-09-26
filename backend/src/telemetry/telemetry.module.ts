import { Module } from '@nestjs/common';
import { TelemetryReceiver } from './telemetry.receiver.js';
import { TelemetryParser } from './telemetry.parser.js';
import { TelemetryHistory } from './telemetry-history.js';

@Module({
  providers: [
    TelemetryReceiver,
    TelemetryParser,
    TelemetryHistory,
  ],
  exports: [
    TelemetryHistory,
  ],
})
export class TelemetryModule {}