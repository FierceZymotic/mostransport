import { Controller, Get, Param, Query } from '@nestjs/common';
import { PredictionService } from './prediction.service.js';
import { ScheduleRepository } from './schedule.repository.js';
import { TripMatcherService } from './trip-matcher.service.js';

@Controller('prediction')
export class PredictionController {
  constructor(
  private readonly predictionService: PredictionService,
  private readonly scheduleRepository: ScheduleRepository,
  private readonly tripMatcherService: TripMatcherService,
) {}

@Get('run/:unitId')
async run(@Param('unitId') unitId: string) {
  return this.predictionService.predictForVehicle(
    Number(unitId),
  );
}

  @Get('db-test')
  async dbTest() {
    return this.predictionService.testDatabase();
  }

  @Get('schedule-test')
  async scheduleTest(
    @Query('trId') trId: string,
    @Query('time') time: string,
  ) {
    try {
      return await this.scheduleRepository.findTargetAction(
        trId,
        new Date(time),
      );
    } catch (error) {
      console.error('SCHEDULE TEST ERROR:', error);
      throw error;
    }
  }
  @Get('matcher-test')
async matcherTest(
  @Query('lat') lat: string,
  @Query('lon') lon: string,
) {
  return this.tripMatcherService.findTrip(
    Number(lat),
    Number(lon),
  );
}
@Get('run-at/:unitId')
async runAt(
  @Param('unitId') unitId: string,
  @Query('time') time: string,
) {
  return this.predictionService.predictForVehicleAt(
    Number(unitId),
    new Date(time),
  );
}
}