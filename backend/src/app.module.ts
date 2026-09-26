import { Module } from '@nestjs/common';

import { TelemetryModule } from './telemetry/telemetry.module.js';
import { PredictionModule } from './prediction/prediction.module.js';

import { PrismaModule } from './prisma/prisma.module.js';

@Module({
  imports: [
    TelemetryModule,
    PredictionModule,
    PrismaModule,
  ],
})
export class AppModule {}