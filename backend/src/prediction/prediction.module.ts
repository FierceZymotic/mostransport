import { Module } from '@nestjs/common';

import { PredictionController } from './prediction.controller.js';
import { PredictionService } from './prediction.service.js';
import { MlClientService } from './ml/ml.client.service.js';
import { TelemetryModule } from '../telemetry/telemetry.module.js';
import { ScheduleRepository } from './schedule.repository.js';
import { TripMatcherService } from './trip-matcher.service.js';
import { LivePredictionService } from './live-prediction.service.js';

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
    ScheduleRepository,
    TripMatcherService,
    LivePredictionService,
  ],

  exports: [
    PredictionService,
    MlClientService,
    ScheduleRepository,
    TripMatcherService,
  ],
})
export class PredictionModule {}