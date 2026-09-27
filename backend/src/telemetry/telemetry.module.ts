import { Module } from '@nestjs/common';
import { TelemetryReceiver } from './telemetry.receiver.js';
import { TelemetryParser } from './telemetry.parser.js';
import { TelemetryHistory } from './telemetry-history.js';
import { TelemetryRepository } from './telemetry.repository.js';
import { TelemetryStreamService } from './telemetry-stream.service.js';
import { TelemetryGateway } from './telemetry.gateway.js';

@Module({
  providers: [
    TelemetryReceiver,
    TelemetryParser,
    TelemetryHistory,
    TelemetryRepository,
    TelemetryStreamService,
    TelemetryGateway,
  ],
  exports: [
    TelemetryHistory,
    TelemetryRepository,
    TelemetryStreamService,
  ],
})
export class TelemetryModule {}