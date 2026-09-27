import { describe, expect, it, vi } from 'vitest';

import { PredictionIneligibleError, PredictionService } from '../src/prediction/prediction.service.js';

const T = new Date('2026-01-06T12:00:00Z');
const state = (offset: number, valid = true, lat = 55.7, lon = 37.6) => ({
  unitId: 7, timestamp: Math.floor(T.getTime() / 1000) + offset, longitude: lon, latitude: lat, locationValid: valid,
  speed: 12, speedMax: 20, course: 0, track: 0, altitude: 0, nsat: 0, pdop: 0,
});
const action = { target_action_id: '1', target_time_begin: '2026-01-06T12:12:00.000Z', time_fact_begin: null, target_lat: 55.71, target_lon: 37.61, manual_fill: false };

function build(overrides: Record<string, any> = {}) {
  const deps = {
    ml: { predict: vi.fn(async (req: any) => ({ request_id: req.request_id, status: 'success', prediction: { delay_seconds: 42, target_time: '2026-01-06T12:12:42Z', reason: null as string | null }, generated_at: '2026-01-06T12:00:01Z', model_version: 'm', feature_schema_version: 'runtime-safe-v1' })) },
    telemetry: { findHistory: vi.fn(async () => [state(-30), state(-10, false, 1, 1)]) },
    prisma: { predictions: { create: vi.fn(async (_args: any) => ({})) } },
    schedule: { findTargetAction: vi.fn(async () => ({ status: 'ok', action, candidates: 1 })), getCurrentDeviation: vi.fn(async () => ({ seconds: 0, status: 'unavailable_no_fact_source' })) },
    matcher: { findTrip: vi.fn(async () => ({ trId: '131672', targetActionId: '1', distanceMeters: 5 })) },
    ...overrides,
  };
  const service = new PredictionService(deps.ml as any, deps.telemetry as any, deps.prisma as any, deps.schedule as any, deps.matcher as any);
  return { service, deps };
}

describe('PredictionService current deviation state', () => {
  it('sends the Contract scalar 0 but records "unavailable" (never a confirmed on-time) without a fact source', async () => {
    const { service, deps } = build();
    const outcome = await service.predictWithStatus(7, T);
    expect(deps.ml.predict.mock.calls[0][0].schedule_context.current_deviation_seconds).toBe(0);
    expect(outcome.currentDeviationStatus).toBe('unavailable_no_fact_source');
    expect(deps.prisma.predictions.create.mock.calls[0][0].data.reason).toBe('degraded:current_deviation_unavailable(no_fact_source)');
  });

  it('does not mark predictions backed by a confirmed fact, nor a known "no fact yet" 0 in replay', async () => {
    for (const [seconds, status] of [[95, 'confirmed_fact'], [0, 'no_fact_before_t']] as const) {
      const { service, deps } = build({ schedule: { findTargetAction: vi.fn(async () => ({ status: 'ok', action, candidates: 1 })), getCurrentDeviation: vi.fn(async () => ({ seconds, status })) } });
      const outcome = await service.predictWithStatus(7, T);
      expect(deps.ml.predict.mock.calls[0][0].schedule_context.current_deviation_seconds).toBe(seconds);
      expect(outcome.currentDeviationStatus).toBe(status);
      expect(deps.prisma.predictions.create.mock.calls[0][0].data.reason).toBeNull();
    }
  });

  it('keeps an ML-provided reason and returns the unmodified ML response from predictForVehicleAt', async () => {
    const { service, deps } = build();
    deps.ml.predict.mockImplementationOnce(async (req: any) => ({ request_id: req.request_id, status: 'success', prediction: { delay_seconds: 1, target_time: '2026-01-06T12:12:01Z', reason: 'ml-reason' }, generated_at: '2026-01-06T12:00:01Z', model_version: 'm', feature_schema_version: 'runtime-safe-v1' }));
    const response = await service.predictForVehicleAt(7, T);
    expect(response.prediction.reason).toBe('ml-reason');
    expect(deps.prisma.predictions.create.mock.calls[0][0].data.reason).toBe('ml-reason');
  });
});

describe('PredictionService trip and target resolution', () => {
  it('matches on the latest strict-valid GPS with the unit identity and the explicit prediction time', async () => {
    const { service, deps } = build();
    await service.predictForVehicleAt(7, T);
    // the latest packet is invalid (and carries other coordinates): it must not drive matching
    expect(deps.matcher.findTrip).toHaveBeenCalledWith(55.7, 37.6, '7', T);
    expect(deps.telemetry.findHistory).toHaveBeenCalledWith(7, T);
    expect(deps.schedule.findTargetAction).toHaveBeenCalledWith('131672', T);
    const req = deps.ml.predict.mock.calls[0][0];
    expect(req.prediction_time).toBe(T.toISOString());
    expect(req.telemetry.map((p: any) => p.lat)).toEqual([55.7, null]); // invalid fix sent without coordinates
  });

  it('returns explicit ineligibility for an ambiguous target instead of choosing an id', async () => {
    const { service, deps } = build({ schedule: { findTargetAction: vi.fn(async () => ({ status: 'ambiguous', action: null, candidates: 2 })), getCurrentDeviation: vi.fn() } });
    const error = await service.predictForVehicleAt(7, T).catch((e) => e);
    expect(error).toBeInstanceOf(PredictionIneligibleError);
    expect(error.code).toBe('TARGET_AMBIGUOUS');
    expect(error.getStatus()).toBe(422);
    expect(deps.ml.predict).not.toHaveBeenCalled();
    expect(deps.prisma.predictions.create).not.toHaveBeenCalled();
  });

  it('returns explicit ineligibility when no target is in the horizon or no trip matches', async () => {
    const none = build({ schedule: { findTargetAction: vi.fn(async () => ({ status: 'none', action: null, candidates: 0 })), getCurrentDeviation: vi.fn() } });
    await expect(none.service.predictForVehicleAt(7, T)).rejects.toMatchObject({ code: 'NO_TARGET_IN_HORIZON' });
    const noTrip = build({ matcher: { findTrip: vi.fn(async () => null) } });
    await expect(noTrip.service.predictForVehicleAt(7, T)).rejects.toMatchObject({ code: 'NO_TRIP_MATCH' });
  });

  it('returns explicit ineligibility when no strict-valid GPS or no telemetry exists', async () => {
    const invalidOnly = build({ telemetry: { findHistory: vi.fn(async () => [state(-5, false)]) } });
    await expect(invalidOnly.service.predictForVehicleAt(7, T)).rejects.toMatchObject({ code: 'NO_VALID_GPS' });
    expect(invalidOnly.deps.matcher.findTrip).not.toHaveBeenCalled();
    const empty = build({ telemetry: { findHistory: vi.fn(async () => []) } });
    await expect(empty.service.predictForVehicleAt(7, T)).rejects.toMatchObject({ code: 'NO_TELEMETRY' });
  });

  it('rejects an invalid explicit prediction time', async () => {
    const { service } = build();
    await expect(service.predictForVehicleAt(7, new Date('bad'))).rejects.toThrow(/valid ISO-8601/);
  });
});
