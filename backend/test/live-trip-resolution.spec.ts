import { ServiceUnavailableException } from '@nestjs/common';
import { describe, expect, it, vi } from 'vitest';

import { LivePredictionService } from '../src/prediction/live-prediction.service.js';
import { PredictionController } from '../src/prediction/prediction.controller.js';
import { PredictionIneligibleError, PredictionService } from '../src/prediction/prediction.service.js';
import { TelemetryRepository } from '../src/telemetry/telemetry.repository.js';
import type { VehicleState } from '../src/telemetry/vehicle-state.js';

// Mapped organizer vehicle (vehicles.current_tr_id) vs. a synthetic emulator unit without identity.
const MAPPED_UNIT = 1105498;
const MAPPED_TRIP = '134040';
const EMULATOR_UNIT = 1001;
const T_SECONDS = Math.floor(Date.parse('2026-01-06T12:00:00Z') / 1000);
const T = new Date(T_SECONDS * 1000);

const packet = (unitId: number, timestamp = T_SECONDS, locationValid = true): VehicleState => ({
  unitId, timestamp, longitude: 37.6, latitude: 55.7, locationValid,
  speed: 18, speedMax: 20, course: 0, track: 0, altitude: 0, nsat: 0, pdop: 0,
});
const action = { target_action_id: '1', target_time_begin: '2026-01-06T12:12:00.000Z', time_fact_begin: null, target_lat: 55.71, target_lon: 37.61, manual_fill: false };

function build(options: { ml?: (req: any) => Promise<any>; target?: any } = {}) {
  const matcher = {
    // The real matcher resolves mapped units to their trip; unmapped synthetic units find no trip here.
    findTrip: vi.fn(async (_lat: number, _lon: number, unitId: string | undefined, _t: Date) =>
      unitId === String(MAPPED_UNIT) ? { trId: MAPPED_TRIP, targetActionId: '1', distanceMeters: 40 } : null),
  };
  const telemetry = {
    findHistory: vi.fn(async (unitId: number) => [packet(unitId, T_SECONDS - 30), packet(unitId, T_SECONDS)]),
    findLatestValidPosition: vi.fn(async () => ({ latitude: 55.7, longitude: 37.6 })),
  };
  const ml = {
    predict: vi.fn(options.ml ?? (async (req: any) => ({
      request_id: req.request_id, status: 'success',
      prediction: { delay_seconds: 11, target_time: '2026-01-06T12:12:11Z', reason: null as string | null },
      generated_at: '2026-01-06T12:00:01Z', model_version: 'hgb-h0-runtime-safe-v1-group-a-v1', feature_schema_version: 'runtime-safe-v1',
    }))),
  };
  const schedule = {
    findTargetAction: vi.fn(async () => options.target ?? { status: 'ok', action, candidates: 1 }),
    getCurrentDeviation: vi.fn(async () => ({ seconds: 0, status: 'unavailable_no_fact_source' })),
  };
  const lastStates = [MAPPED_UNIT, EMULATOR_UNIT].map((unit) => ({
    unit_id: String(unit), tr_id: null, timestamp: T, latitude: 55.7, longitude: 37.6, speed: 18, location_valid: true,
  }));
  const prisma = {
    predictions: { create: vi.fn(async (_args: any) => ({})), findMany: vi.fn(async () => []) },
    vehicle_last_state: { findMany: vi.fn(async () => lastStates) },
  };
  const service = new PredictionService(ml as any, telemetry as any, prisma as any, schedule as any, matcher as any);
  const controller = new PredictionController(service, schedule as any, matcher as any, prisma as any);
  let subscriber: (state: VehicleState) => void = () => undefined;
  const stream = {
    subscribeState: vi.fn((fn: (state: VehicleState) => void) => { subscriber = fn; return () => undefined; }),
    publishPrediction: vi.fn(),
  };
  const live = new LivePredictionService(stream as any, service);
  live.onModuleInit();
  const sendPacket = async (state: VehicleState) => {
    subscriber(state);
    await new Promise((r) => setTimeout(r, 0));
  };
  return { service, controller, matcher, telemetry, ml, prisma, stream, sendPacket };
}

const routeOf = (rows: { id: string; route: string | null }[], unit: number) => rows.find((row) => row.id === String(unit))?.route;

describe('live trip resolution (one canonical path for dashboard and prediction)', () => {
  it('resolves a mapped vehicle to its real trip from telemetry (VehicleState -> latest valid GPS -> matcher)', async () => {
    const { service, matcher, telemetry } = build();
    const trip = await service.resolveTrip(MAPPED_UNIT, T);
    expect(trip).toEqual({ status: 'resolved', match: { trId: MAPPED_TRIP, targetActionId: '1', distanceMeters: 40 } });
    expect(telemetry.findLatestValidPosition).toHaveBeenCalledWith(MAPPED_UNIT, T);
    expect(matcher.findTrip).toHaveBeenCalledWith(55.7, 37.6, String(MAPPED_UNIT), T);
  });

  it('uses the same real trId for the dashboard summary and the live prediction request', async () => {
    const { controller, ml, stream, sendPacket, prisma } = build();
    await sendPacket(packet(MAPPED_UNIT));
    const request = ml.predict.mock.calls[0][0];
    expect(request.vehicle_context.tr_id).toBe(MAPPED_TRIP);
    expect(prisma.predictions.create.mock.calls[0][0].data.tr_id).toBe(MAPPED_TRIP);
    expect(stream.publishPrediction).toHaveBeenCalledTimes(1);
    const summary = await controller.dashboardSummary();
    expect(routeOf(summary.vehicles, MAPPED_UNIT)).toBe(MAPPED_TRIP);
    expect(routeOf(await controller.dashboardVehicles(), MAPPED_UNIT)).toBe(MAPPED_TRIP);
  });

  it('keeps the known trip id on the dashboard when no prediction exists (ML unavailable)', async () => {
    const { controller, ml, sendPacket, prisma } = build({
      ml: async () => { throw new ServiceUnavailableException('ML service not ready'); },
    });
    await sendPacket(packet(MAPPED_UNIT));
    expect(ml.predict).toHaveBeenCalledTimes(1);
    expect(prisma.predictions.create).not.toHaveBeenCalled();
    const summary = await controller.dashboardSummary();
    expect(routeOf(summary.vehicles, MAPPED_UNIT)).toBe(MAPPED_TRIP); // not dependent on prediction history
  });

  it('an unresolvable vehicle gets no fake trip: route null (never "LIVE"), prediction stays NO_TRIP_MATCH', async () => {
    const { controller, service, ml, sendPacket, stream } = build();
    await expect(service.predictWithStatus(EMULATOR_UNIT, T)).rejects.toMatchObject({ code: 'NO_TRIP_MATCH' });
    await sendPacket(packet(EMULATOR_UNIT));
    expect(ml.predict).not.toHaveBeenCalled();
    expect(stream.publishPrediction).not.toHaveBeenCalled();
    const summary = await controller.dashboardSummary();
    expect(routeOf(summary.vehicles, EMULATOR_UNIT)).toBeNull();
    expect(JSON.stringify(summary)).not.toContain('LIVE');
    expect(routeOf(await controller.dashboardVehicles(), EMULATOR_UNIT)).toBeNull();
  });

  it('no strict-valid GPS or a failing lookup also yields null, not a placeholder', async () => {
    const noFix = build();
    noFix.telemetry.findLatestValidPosition.mockResolvedValue(null as any);
    expect(await noFix.service.resolveTrip(MAPPED_UNIT, T)).toEqual({ status: 'NO_VALID_GPS', match: null });
    expect(routeOf((await noFix.controller.dashboardSummary()).vehicles, MAPPED_UNIT)).toBeNull();
    expect(noFix.matcher.findTrip).not.toHaveBeenCalled();

    const broken = build();
    broken.matcher.findTrip.mockRejectedValue(new Error('db down'));
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    expect(routeOf((await broken.controller.dashboardSummary()).vehicles, MAPPED_UNIT)).toBeNull();
    warn.mockRestore();
  });

  it('keeps the target-ambiguity refusal: prediction refused, trip identity still shown', async () => {
    const { controller, service, ml } = build({ target: { status: 'ambiguous', action: null, candidates: 2 } });
    const error = await service.predictWithStatus(MAPPED_UNIT, T).catch((e) => e);
    expect(error).toBeInstanceOf(PredictionIneligibleError);
    expect(error.code).toBe('TARGET_AMBIGUOUS');
    expect(ml.predict).not.toHaveBeenCalled();
    expect(routeOf((await controller.dashboardSummary()).vehicles, MAPPED_UNIT)).toBe(MAPPED_TRIP);
  });

  it('live prediction resolves the trip from its own Contract history (same anchor, no second lookup)', async () => {
    const { service, telemetry, matcher } = build();
    await service.predictWithStatus(MAPPED_UNIT, T);
    expect(telemetry.findLatestValidPosition).not.toHaveBeenCalled();
    expect(matcher.findTrip).toHaveBeenCalledWith(55.7, 37.6, String(MAPPED_UNIT), T);
  });
});

describe('TelemetryRepository.findLatestValidPosition', () => {
  it('returns the latest strict-valid fix <= T in (timestamp, id) order, skipping non-finite coordinates', async () => {
    const pages = [
      [{ latitude: Number.NaN, longitude: 37.6 }, ...Array.from({ length: 49 }, () => ({ latitude: Number.NaN, longitude: Number.NaN }))],
      [{ latitude: 55.71, longitude: 37.61 }],
    ];
    const findMany = vi.fn(async (_args: any) => pages.shift() ?? []);
    const repository = new TelemetryRepository({ telemetry: { findMany } } as any);
    expect(await repository.findLatestValidPosition(MAPPED_UNIT, T)).toEqual({ latitude: 55.71, longitude: 37.61 });
    expect(findMany.mock.calls[0][0]).toMatchObject({
      where: { unit_id: String(MAPPED_UNIT), timestamp: { lte: T }, location_valid: true },
      orderBy: [{ timestamp: 'desc' }, { id: 'desc' }],
      skip: 0,
    });
    expect(findMany.mock.calls[1][0].skip).toBe(50);
  });

  it('returns null when the unit has no valid fix at or before T', async () => {
    const repository = new TelemetryRepository({ telemetry: { findMany: vi.fn(async () => []) } } as any);
    expect(await repository.findLatestValidPosition(EMULATOR_UNIT, T)).toBeNull();
  });
});
