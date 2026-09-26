import { Controller, Get, Param } from '@nestjs/common';
import { PredictionService } from './prediction.service.js';

@Controller('prediction')
export class PredictionController {
  constructor(
    private readonly predictionService: PredictionService,
  ) {}

  @Get('run/:unitId')
  async run(@Param('unitId') unitId: string) {
    return this.predictionService.predictForVehicle(
      Number(unitId),
    );
  }
}