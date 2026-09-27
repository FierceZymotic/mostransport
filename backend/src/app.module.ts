import { Module } from '@nestjs/common';

import { AppController } from './app.controller.js';
import { AppService } from './app.service.js';
import { TelemetryModule } from './telemetry/telemetry.module.js';
import { PredictionModule } from './prediction/prediction.module.js';

import { PrismaModule } from './prisma/prisma.module.js';

@Module({
  imports: [
    TelemetryModule,
    PredictionModule,
    PrismaModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}