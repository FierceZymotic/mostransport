import { Module } from '@nestjs/common';
import { PredictionController } from './prediction.controller.js';
import { PredictionService } from './prediction.service.js';
import { MlClientService } from './ml/ml.client.service.js';
import { TelemetryModule } from '../telemetry/telemetry.module.js';

@Module({
  imports: [
    TelemetryModule,
  ],
  controllers: [
    PredictionController,
  ],
  providers: [
    PredictionService,
    MlClientService,
  ],
  exports: [
    PredictionService,
    MlClientService,
  ],
})
export class PredictionModule {}