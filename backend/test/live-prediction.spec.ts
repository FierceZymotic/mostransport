import { describe, expect, it, vi } from 'vitest';

import { LivePredictionService } from '../src/prediction/live-prediction.service.js';
import { PredictionIneligibleError } from '../src/prediction/prediction.service.js';
import type { VehicleState } from '../src/telemetry/vehicle-state.js';

const packet = (unitId: number, timestamp: number, locationValid = true): VehicleState => ({
  unitId, timestamp, longitude: 37.6, latitude: 55.7, locationValid,
  speed: 10, speedMax: 10, course: 0, track: 0, altitude: 0, nsat: 0, pdop: 0,
});

function setup(predict: (unitId: number, at: Date) => Promise<any>) {
  const outcome = async (unitId: number, at: Date) => ({ response: await predict(unitId, at), currentDeviationStatus: 'unavailable_no_fact_source' });
  let subscriber: (state: VehicleState) => void = () => undefined;
  const stream = {
    subscribeState: vi.fn((fn: (state: VehicleState) => void) => { subscriber = fn; return () => undefined; }),
    publishPrediction: vi.fn(),
  };
  const predictionService = { predictWithStatus: vi.fn(outcome) };
  const live = new LivePredictionService(stream as any, predictionService as any);
  live.onModuleInit();
  const send = async (state: VehicleState) => {
    subscriber(state);
    await new Promise((r) => setTimeout(r, 0));
  };
  return { stream, predictionService, send };
}

describe('LivePredictionService', () => {
  it('predicts at the packet event time and publishes the prediction', async () => {
    const response = { request_id: 'r', prediction: { delay_seconds: 30 } };
    const { stream, predictionService, send } = setup(async () => response);
    await send(packet(1105498, 1767700000));
    expect(predictionService.predictWithStatus).toHaveBeenCalledWith(1105498, new Date(1767700000 * 1000));
    expect(stream.publishPrediction).toHaveBeenCalledWith(response, { current_deviation_status: 'unavailable_no_fact_source' });
  });

  it('logs and skips an ineligible point (e.g. ambiguous target) for the cycle without publishing or crashing', async () => {
    const { stream, predictionService, send } = setup(async () => {
      throw new PredictionIneligibleError('TARGET_AMBIGUOUS', 'tie');
    });
    await send(packet(7, 1767700000));
    await send(packet(7, 1767700005)); // same 60 s cycle: not retried
    await send(packet(7, 1767700061)); // next cycle: tried again
    expect(predictionService.predictWithStatus).toHaveBeenCalledTimes(2);
    expect(stream.publishPrediction).not.toHaveBeenCalled();
  });

  it('retries a transient failure on the next valid packet and ignores invalid-GPS packets', async () => {
    let calls = 0;
    const { stream, predictionService, send } = setup(async () => {
      calls += 1;
      if (calls === 1) throw new Error('ML service is not ready');
      return { request_id: 'r', prediction: { delay_seconds: 1 } };
    });
    await send(packet(7, 1767700000));
    await send(packet(7, 1767700003, false)); // invalid GPS never triggers a prediction
    await send(packet(7, 1767700005));
    expect(predictionService.predictWithStatus).toHaveBeenCalledTimes(2);
    expect(stream.publishPrediction).toHaveBeenCalledTimes(1);
  });
});
