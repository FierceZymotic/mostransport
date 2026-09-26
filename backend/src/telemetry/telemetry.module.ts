import { Module } from '@nestjs/common';
import { TelemetryReceiver } from './telemetry.receiver.js';
import { TelemetryParser } from './telemetry.parser.js';
import { TelemetryHistory } from './telemetry-history.js';
import { TelemetryRepository } from './telemetry.repository.js';

@Module({
  providers: [
    TelemetryReceiver,
    TelemetryParser,
    TelemetryHistory,
    TelemetryRepository,
  ],
  exports: [
    TelemetryHistory,
    TelemetryRepository,
  ],
})
export class TelemetryModule {}