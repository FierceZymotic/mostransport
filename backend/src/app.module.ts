import { Module } from '@nestjs/common';

import { TelemetryModule } from './telemetry/telemetry.module.js';
import { PredictionModule } from './prediction/prediction.module.js';

@Module({
  imports: [
    TelemetryModule,
    PredictionModule,
  ],
})
export class AppModule {}